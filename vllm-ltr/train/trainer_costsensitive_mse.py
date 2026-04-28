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


def compute_midpoints(boundaries, num_classes, max_length=8192):
    midpoints = []
    for c in range(num_classes):
        idx = num_classes - 1 - c
        if idx == 0:
            lo, hi = 0, boundaries[0]
        elif idx == len(boundaries):
            lo, hi = boundaries[-1], max_length
        else:
            lo, hi = boundaries[idx - 1], boundaries[idx]
        midpoints.append((lo + hi) / 2.0)
    return midpoints


class PercentileDataset(Dataset):
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

    # Compute percentile boundaries from training data only
    boundaries, actual_num_classes = compute_percentile_boundaries(train_lengths, args.num_classes)

    print(f"Requested {args.num_classes} classes, got {actual_num_classes} after dedup")
    print(f"Percentile boundaries: {boundaries}")
    print(f"Length stats - min: {train_lengths.min()}, max: {train_lengths.max()}, "
          f"mean: {train_lengths.mean():.1f}, median: {np.median(train_lengths):.1f}")

    # Print class distribution
    train_labels = np.array([actual_num_classes - 1 - int(np.searchsorted(boundaries, l)) for l in train_lengths])
    for c in range(actual_num_classes):
        count = (train_labels == c).sum()
        # map class back to length range
        idx = actual_num_classes - 1 - c
        if idx == 0:
            lo, hi = 0, int(boundaries[0])
        elif idx == len(boundaries):
            lo, hi = int(boundaries[-1]), int(train_lengths.max())
        else:
            lo, hi = int(boundaries[idx - 1]), int(boundaries[idx])
        print(f"  Class {c}: {count} samples ({count/len(train_labels)*100:.1f}%) — tokens [{lo}, {hi})")

    # Compute midpoints and cost matrix for cost-sensitive loss
    midpoints = compute_midpoints(boundaries, actual_num_classes, args.label_max_length)
    print(f"Class midpoints (tokens): {[f'{m:.0f}' for m in midpoints]}")
    midpoints_tensor = torch.tensor(midpoints, dtype=torch.float32, device='cuda')
    print(f"Midpoint range: {midpoints_tensor.min().item():.0f} - {midpoints_tensor.max().item():.0f} tokens")

    config.model.num_labels = actual_num_classes
    print("num_labels:", config.model.num_labels)

    with set_default_torch_dtype(torch.float32):
        with torch.device('cuda'):
            predictor = prefill_predictor_model(pred_model=config.model.pred_model, num_labels=config.model.num_labels, mtype=config.model.mtype, activation=config.model.activation, max_length=config.model.max_length, max_batch_size=config.model.max_batch_size)

    train_dataset = PercentileDataset(train_data, llama3_tokenizer, boundaries, max_length=config.model.max_length, precomputed_lengths=train_lengths)
    test_dataset = PercentileDataset(test_data, llama3_tokenizer, boundaries, max_length=config.model.max_length, precomputed_lengths=test_lengths)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(predictor.model.parameters(), lr=args.lr, weight_decay=args.wc)
    optimizer.zero_grad()

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

                labels = labels.reshape(1, -1)
                labels = labels.to("cuda")
                assert labels.max().item() < predictor.model.num_labels, \
                    f"Label {labels.max().item()} >= num_labels {predictor.model.num_labels}"
                logits = outputs.view(-1, predictor.model.num_labels)

                # Token-space MSE: same as class-index MSE but with
                # midpoints instead of arange. A 1-class error at the
                # long end (~2400 tokens) is penalized ~100x more than
                # at the short end (~8 tokens).
                p = logits.softmax(dim=-1)
                pred_tokens = p @ midpoints_tensor          # expected token count
                true_tokens = midpoints_tensor[labels.view(-1)]  # midpoint of true class
                loss = F.mse_loss(pred_tokens, true_tokens)

            if args.print_loss:
                print("loss: ", loss )
            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            total_loss += loss.item()
            idx += 1
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")

        true_labels = []
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

                predicted_scores = outputs.argmax(dim=-1).tolist()

                true_labels.extend(labels.tolist())
                predictions.extend(predicted_scores)

            tau, score = kendalltau(true_labels, predictions)
            print(f"Kendall's Tau: {tau}, p-value: {score}")

            true_arr = np.array(true_labels)
            pred_arr = np.array(predictions)
            print(f"Exact accuracy: {(true_arr == pred_arr).sum() / len(true_arr):.4f}")
            print(f"Within-1 accuracy: {(np.abs(true_arr - pred_arr) <= 1).sum() / len(true_arr):.4f}")

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
        "loss": "costsensitive_mse",
        "midpoints": midpoints,
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