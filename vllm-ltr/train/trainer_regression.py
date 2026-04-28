from transformers import AutoTokenizer, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
import evaluate, datasets
import numpy as np
from vllm.model_executor import prefill_predictor
from vllm.config_predictor import PrefillPredictorConfig
from vllm.model_executor.prefill_predictor import prefill_predictor_model
import json
import torch
import torch.nn.functional as F
from argparse import ArgumentParser, Namespace
from vllm.model_executor.model_loader.utils import set_default_torch_dtype
from scipy.stats import kendalltau
from allrank.utils.file_utils import create_output_dirs, PathsContainer, copy_local_to_gs
import os
from tqdm import tqdm
import math

def parse_args():
    parser = ArgumentParser("allRank")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--print-loss", action='store_true')
    parser.add_argument("--file", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wc", type=float, default=0.01)
    parser.add_argument("--job-dir", type=str, required=True)
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--label-max-length", type=int, default=8192)
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--log-space", action='store_true',
                        help="Train in log(1+tokens) space instead of raw token space")
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Meta-Llama-3-70B")

    return parser.parse_args()


def compute_lengths(data, tokenizer):
    lengths = []
    for item in tqdm(data, desc="Tokenizing outputs for length computation"):
        length = len(tokenizer(item['generated'])['input_ids'])
        lengths.append(length)
    return np.array(lengths)


def compute_percentile_boundaries(lengths, num_classes):
    percentiles = np.linspace(0, 100, num_classes + 1)[1:-1]
    boundaries = np.percentile(lengths, percentiles)
    boundaries = np.unique(boundaries)
    actual_num_classes = len(boundaries) + 1
    return boundaries, actual_num_classes


class RegressionDataset(Dataset):
    def __init__(self, data, tokenizer, boundaries, max_length=2048, precomputed_lengths=None):
        self.data = data
        self.tokenizer = tokenizer
        self.boundaries = boundaries
        self.max_length = max_length
        self.num_classes = len(boundaries) + 1
        self.precomputed_lengths = precomputed_lengths

    def __len__(self):
        return len(self.data)

    def __len2label__(self, length):
        # higher label = shorter output following the authors convention
        return self.num_classes - 1 - int(np.searchsorted(self.boundaries, length))

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        if self.precomputed_lengths is not None:
            origin_len = int(self.precomputed_lengths[idx])
        else:
            origin_len = len(self.tokenizer(item['generated'])['input_ids'])
        label = self.__len2label__(origin_len)
        return prompt, label, origin_len


def run():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()

    llama3_tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    prefill_predictor_model_config = args.config
    config = PrefillPredictorConfig.from_json(prefill_predictor_model_config)

    dataset_path = args.file
    dataset = []

    with open(dataset_path) as f:
        for jsonObj in f:
            info = json.loads(jsonObj)
            dataset.append(info)

    # First pass: compute all output lengths
    print("=== Computing output lengths (first pass) ===")
    all_lengths = compute_lengths(dataset, llama3_tokenizer)

    # 90/10 split
    split_idx = int(0.9 * len(dataset))
    train_data = dataset[:split_idx]
    test_data = dataset[split_idx:]
    train_lengths = all_lengths[:split_idx]
    test_lengths = all_lengths[split_idx:]

    # Compute percentile boundaries from training data (for eval binning)
    boundaries, actual_num_classes = compute_percentile_boundaries(train_lengths, args.num_classes)

    print(f"Requested {args.num_classes} classes, got {actual_num_classes} after dedup")
    print(f"Percentile boundaries: {boundaries}")
    print(f"Length stats - min: {train_lengths.min()}, max: {train_lengths.max()}, "
          f"mean: {train_lengths.mean():.1f}, median: {np.median(train_lengths):.1f}")

    # Print class distribution (for reference)
    train_labels = np.array([actual_num_classes - 1 - int(np.searchsorted(boundaries, l)) for l in train_lengths])
    for c in range(actual_num_classes):
        count = (train_labels == c).sum()
        idx_b = actual_num_classes - 1 - c
        if idx_b == 0:
            lo, hi = 0, int(boundaries[0])
        elif idx_b == len(boundaries):
            lo, hi = int(boundaries[-1]), int(train_lengths.max())
        else:
            lo, hi = int(boundaries[idx_b - 1]), int(boundaries[idx_b])
        print(f"  Class {c}: {count} samples ({count/len(train_labels)*100:.1f}%) — tokens [{lo}, {hi})")

    if args.log_space:
        print("Training in log(1 + tokens) space")
    else:
        print("Training in raw token space")

    # Single regression output
    config.model.num_labels = 1
    print("num_labels:", config.model.num_labels)

    with set_default_torch_dtype(torch.float32):
        with torch.device('cuda'):
            predictor = prefill_predictor_model(pred_model=config.model.pred_model, num_labels=config.model.num_labels, mtype=config.model.mtype, activation=config.model.activation, max_length=config.model.max_length, max_batch_size=config.model.max_batch_size)

    train_dataset = RegressionDataset(train_data, llama3_tokenizer, boundaries, max_length=config.model.max_length, precomputed_lengths=train_lengths)
    test_dataset = RegressionDataset(test_data, llama3_tokenizer, boundaries, max_length=config.model.max_length, precomputed_lengths=test_lengths)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(predictor.model.parameters(), lr=args.lr, weight_decay=args.wc)
    optimizer.zero_grad()

    # Helper to convert token length to regression target
    def to_target(token_lengths):
        t = token_lengths.float()
        if args.log_space:
            t = torch.log1p(t)
        return t

    # Helper to convert regression prediction back to token space
    def from_prediction(pred):
        if args.log_space:
            pred = torch.expm1(pred)
        return pred.clamp(min=0)

    # Helper to bin predicted tokens into percentile classes
    def tokens_to_class(token_lengths_np):
        classes = []
        for t in token_lengths_np:
            idx_b = int(np.searchsorted(boundaries, t))
            c = actual_num_classes - 1 - idx_b
            c = max(0, min(actual_num_classes - 1, c))
            classes.append(c)
        return np.array(classes)

    for epoch in range(args.epoch):
        predictor.model.train()
        total_loss = 0
        idx = 0
        for prompt, labels, origin_len in tqdm(train_dataloader):
            prompt = list(prompt)

            encoded_inputs = predictor.tokenizer(prompt, max_length=config.model.max_length, padding=True, truncation=True, return_tensors="pt")

            input_ids = encoded_inputs['input_ids'].to("cuda:0")
            attention_mask = encoded_inputs['attention_mask'].to("cuda:0")

            with torch.autocast(device_type="cuda"):

                outputs = predictor(input_ids, attention_mask)

                # Regression target: true token length (or log)
                target = to_target(origin_len.to("cuda"))
                pred = outputs.view(-1)
                loss = F.mse_loss(pred, target)

            if args.print_loss:
                print("loss: ", loss )
            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            total_loss += loss.item()
            idx += 1
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")

        true_labels = []
        true_tokens_list = []
        pred_tokens_list = []
        predictions = []

        predictor.model.eval()
        with torch.no_grad():
            for prompt, labels, origin_len in tqdm(test_dataloader):
                prompt = list(prompt)

                encoded_inputs = predictor.tokenizer(prompt, max_length=config.model.max_length, padding=True, truncation=True, return_tensors="pt")
                input_ids = encoded_inputs['input_ids'].to("cuda:0")
                attention_mask = encoded_inputs['attention_mask'].to("cuda:0")
                with torch.autocast(device_type="cuda"):
                    outputs = predictor(input_ids, attention_mask)

                # Convert predictions back to token space
                pred_raw = outputs.view(-1)
                pred_tokens = from_prediction(pred_raw).cpu().numpy()

                true_labels.extend(labels.tolist())
                true_tokens_list.extend(origin_len.tolist())
                pred_tokens_list.extend(pred_tokens.tolist())

                # Bin predicted tokens into classes for accuracy comparison
                pred_classes = tokens_to_class(pred_tokens)
                predictions.extend(pred_classes.tolist())

            # Kendall's tau on raw token predictions (ranking quality)
            tau_tokens, p_tokens = kendalltau(true_tokens_list, pred_tokens_list)
            print(f"Kendall's Tau (tokens): {tau_tokens}, p-value: {p_tokens}")

            # Kendall's tau on binned class predictions (comparable to classifiers)
            tau_class, p_class = kendalltau(true_labels, predictions)
            print(f"Kendall's Tau (class): {tau_class}, p-value: {p_class}")

            # Token-level error stats
            true_tok = np.array(true_tokens_list)
            pred_tok = np.array(pred_tokens_list)
            mae = np.mean(np.abs(true_tok - pred_tok))
            rmse = np.sqrt(np.mean((true_tok - pred_tok) ** 2))
            print(f"MAE: {mae:.1f} tokens, RMSE: {rmse:.1f} tokens")

            # Class-level accuracy (binned)
            true_arr = np.array(true_labels)
            pred_arr = np.array(predictions)
            print(f"Exact accuracy (binned): {(true_arr == pred_arr).sum() / len(true_arr):.4f}")
            print(f"Within-1 accuracy (binned): {(np.abs(true_arr - pred_arr) <= 1).sum() / len(true_arr):.4f}")

    paths = PathsContainer.from_args(args.job_dir, args.run_id, prefill_predictor_model_config)

    usage_config_path = os.path.join(paths.output_dir, "usage_config.json")

    finetuned_model_output_path = os.path.join(paths.output_dir, "finetuned")

    config.model.path = str(finetuned_model_output_path)

    create_output_dirs(paths.output_dir)

    PrefillPredictorConfig.to_json(config, usage_config_path)

    # Save percentile boundaries alongside the model
    boundaries_path = os.path.join(paths.output_dir, "percentile_boundaries.json")
    boundaries_info = {
        "boundaries": boundaries.tolist(),
        "num_classes": actual_num_classes,
        "num_classes_requested": args.num_classes,
        "label_max_length": args.label_max_length,
        "convention": "higher_label=shorter_output",
        "loss": "regression_mse" + ("_logspace" if args.log_space else ""),
        "output": "single_token_count" + ("_logspace" if args.log_space else ""),
        "inference_static": "bin predicted tokens into percentile classes",
        "inference_dsrtf": "use raw predicted tokens as est_total",
        "length_stats": {
            "min": int(train_lengths.min()),
            "max": int(train_lengths.max()),
            "mean": float(train_lengths.mean()),
            "median": float(np.median(train_lengths))
        }
    }
    with open(boundaries_path, 'w') as f:
        json.dump(boundaries_info, f, indent=2)
    print(f"Saved percentile boundaries to {boundaries_path}")

    predictor.model.config.__dict__['num_labels'] = config.model.num_labels

    predictor.model = predictor.model.half()
    predictor.model.save_pretrained(finetuned_model_output_path)


if __name__ == "__main__":
    run()