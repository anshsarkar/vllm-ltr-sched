import numpy as np
from scipy.stats import kendalltau, rankdata

from data import (
    PREDICTORS, CLASSIFIERS, RATES,
    UNIFORM_BUCKET_SIZES, PCTL_CLASSIFIERS, LABEL_MAX_LENGTH,
    output_len_to_class_uniform, output_len_to_class_pctl,
)

def kendall_tau_vs_sjf(all_data):
    rows = []
    for sched in PREDICTORS:
        for rate in RATES:
            d = all_data[(sched, rate)]
            sjf = all_data[("sjf", rate)]
            # Both tpt-* and opt-xxx sort by descending aux_score,
            # and SJF sorts by ascending output_len (shortest first).
            # So rank by -aux_scores and rank by output_lens.
            pred_rank = rankdata(-d["aux_scores"])
            true_rank = rankdata(sjf["output_lens"])
            tau, pval = kendalltau(pred_rank, true_rank)
            rows.append({
                "scheduler": sched, "rate": rate,
                "tau": tau, "pvalue": pval,
            })
    return rows

def error_direction_breakdown(all_data):
    rows = []
    for sched in PREDICTORS:
        for rate in RATES:
            d = all_data[(sched, rate)]
            sjf = all_data[("sjf", rate)]
            nlat_delta = d["nlatencies"] - sjf["nlatencies"]
            n = len(nlat_delta)

            # Rank-based for all predictors (including classifiers)
            # Both tpt-* and opt-xxx sort by descending aux_score
            pred_rank = rankdata(-d["aux_scores"])
            true_rank = rankdata(sjf["output_lens"])
            disp = pred_rank - true_rank
            las_mask = disp < 0   # ranked earlier than oracle = long-as-short
            sal_mask = disp > 0   # ranked later than oracle = short-as-long
            correct_mask = disp == 0

            for name, mask in [
                ("long_as_short", las_mask),
                ("correct", correct_mask),
                ("short_as_long", sal_mask),
            ]:
                c = mask.sum()
                rows.append({
                    "scheduler": sched, "rate": rate,
                    "direction": name,
                    "count": int(c),
                    "frac": c / n,
                    "mean_nlatency_delta": float(nlat_delta[mask].mean()) if c > 0 else 0.0,
                })
    return rows


def _get_true_class(output_lens, scheduler, pctl_boundaries):
    """Get true class labels for a classifier given its bucket config."""
    if scheduler in UNIFORM_BUCKET_SIZES:
        return output_len_to_class_uniform(
            output_lens, UNIFORM_BUCKET_SIZES[scheduler]
        )
    elif scheduler in PCTL_CLASSIFIERS and pctl_boundaries:
        return output_len_to_class_pctl(
            output_lens, pctl_boundaries[scheduler]
        )
    else:
        raise ValueError(f"No bucket config for {scheduler}")

def error_concentration(all_data):
    # Compute decile edges from SJF data at any rate (all aligned)
    sjf_ref = all_data[("sjf", RATES[0])]
    input_edges = np.percentile(sjf_ref["input_lens"], np.arange(0, 110, 10))
    output_edges = np.percentile(sjf_ref["output_lens"], np.arange(0, 110, 10))

    result = {}
    for sched in PREDICTORS:
        # Accumulate across all rates
        accum = np.zeros((10, 10))
        counts = np.zeros((10, 10))
        for rate in RATES:
            d = all_data[(sched, rate)]
            sjf = all_data[("sjf", rate)]
            pred_rank = rankdata(-d["aux_scores"])
            true_rank = rankdata(sjf["output_lens"])
            abs_disp = np.abs(pred_rank - true_rank)

            # Bin each request into its decile cell
            i_bin = np.clip(np.digitize(sjf["input_lens"], input_edges[1:-1]), 0, 9)
            o_bin = np.clip(np.digitize(sjf["output_lens"], output_edges[1:-1]), 0, 9)
            for i in range(len(abs_disp)):
                accum[i_bin[i], o_bin[i]] += abs_disp[i]
                counts[i_bin[i], o_bin[i]] += 1

        # Mean per cell
        with np.errstate(divide="ignore", invalid="ignore"):
            result[sched] = np.where(counts > 0, accum / counts, 0)

    return result, input_edges, output_edges

def confusion_matrices(all_data, pctl_boundaries=None):
    result = {}
    for sched in CLASSIFIERS:
        if sched in UNIFORM_BUCKET_SIZES:
            num_classes = _num_classes_uniform(UNIFORM_BUCKET_SIZES[sched])
        else:
            num_classes = 10  # pctl classifiers always have 10

        cm = np.zeros((num_classes, num_classes), dtype=int)
        for rate in RATES:
            d = all_data[(sched, rate)]
            sjf = all_data[("sjf", rate)]
            pred_class = d["aux_scores"].astype(int)
            true_class = _get_true_class(
                sjf["output_lens"], sched, pctl_boundaries
            )
            # Clamp to valid range
            pred_class = np.clip(pred_class, 0, num_classes - 1)
            true_class = np.clip(true_class, 0, num_classes - 1)
            for t, p in zip(true_class, pred_class):
                cm[t, p] += 1

        result[sched] = cm
    return result


def _num_classes_uniform(group_size):
    """Number of classes for a uniform-bucket classifier."""
    return (LABEL_MAX_LENGTH // group_size) + 1


def outlier_overlap(all_data, top_frac=0.05):
    rows = []
    for rate in RATES:
        sjf = all_data[("sjf", rate)]
        n = len(sjf["nlatencies"])
        k = max(1, int(n * top_frac))

        # For each predictor, get the indices of top-k damage requests
        flag_counts = np.zeros(n, dtype=int)
        for sched in PREDICTORS:
            d = all_data[(sched, rate)]
            delta = d["nlatencies"] - sjf["nlatencies"]
            top_idx = np.argsort(delta)[-k:]
            flag_counts[top_idx] += 1

        n_pred = len(PREDICTORS)
        flagged_any = (flag_counts > 0).sum()
        rows.append({
            "rate": rate,
            "n_requests": n,
            "top_k": k,
            "flagged_by_all": int((flag_counts == n_pred).sum()),
            "flagged_by_majority": int((flag_counts > n_pred / 2).sum()),
            "flagged_by_one_only": int((flag_counts == 1).sum()),
            "flagged_any": int(flagged_any),
        })
    return rows

def noise_floor(all_data, n_buckets=10):
    sjf_ref = all_data[("sjf", RATES[0])]
    edges = np.percentile(
        sjf_ref["input_lens"],
        np.linspace(0, 100, n_buckets + 1)
    )

    rows = []
    for sched in PREDICTORS:
        for b in range(n_buckets):
            lo, hi = edges[b], edges[b + 1]
            all_var = []
            all_disp = []
            for rate in RATES:
                sjf = all_data[("sjf", rate)]
                d = all_data[(sched, rate)]
                if b < n_buckets - 1:
                    mask = (sjf["input_lens"] >= lo) & (sjf["input_lens"] < hi)
                else:
                    mask = (sjf["input_lens"] >= lo) & (sjf["input_lens"] <= hi)
                if mask.sum() == 0:
                    continue
                all_var.append(np.var(sjf["output_lens"][mask]))
                pred_rank = rankdata(-d["aux_scores"])
                true_rank = rankdata(sjf["output_lens"])
                all_disp.append(np.mean(np.abs(pred_rank[mask] - true_rank[mask])))

            rows.append({
                "scheduler": sched,
                "bucket_idx": b,
                "bucket_lo": lo,
                "bucket_hi": hi,
                "n_requests": int(mask.sum()) if len(all_var) > 0 else 0,
                "output_len_variance": float(np.mean(all_var)) if all_var else 0.0,
                "mean_abs_displacement": float(np.mean(all_disp)) if all_disp else 0.0,
            })
    return rows
