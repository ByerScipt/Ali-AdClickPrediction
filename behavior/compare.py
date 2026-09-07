import argparse
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from behavior.train import expected_calibration_error, grouped_auc


def compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    tie_start = np.empty(len(values), dtype=bool)
    tie_start[0] = True
    tie_start[1:] = sorted_values[1:] != sorted_values[:-1]
    starts = np.flatnonzero(tie_start)
    ends = np.append(starts[1:], len(values))
    average_rank = 0.5 * (starts + ends - 1) + 1.0
    midrank = np.repeat(average_rank, ends - starts)
    output = np.empty(len(values), dtype=np.float64)
    output[order] = midrank
    return output


def paired_delong(
    labels: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
) -> dict[str, float]:
    positive_order = np.argsort(-labels, kind="stable")
    predictions = np.stack([baseline_prediction, candidate_prediction])[
        :, positive_order
    ]
    positive_count = int(labels.sum())
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("DeLong requires both positive and negative labels")
    positive = predictions[:, :positive_count]
    negative = predictions[:, positive_count:]
    positive_midrank = np.vstack([compute_midrank(row) for row in positive])
    negative_midrank = np.vstack([compute_midrank(row) for row in negative])
    total_midrank = np.vstack([compute_midrank(row) for row in predictions])
    auc = total_midrank[:, :positive_count].sum(axis=1) / (
        positive_count * negative_count
    )
    auc -= (positive_count + 1.0) / (2.0 * negative_count)
    positive_contribution = (
        total_midrank[:, :positive_count] - positive_midrank
    ) / negative_count
    negative_contribution = (
        1.0 - (total_midrank[:, positive_count:] - negative_midrank) / positive_count
    )
    covariance = np.cov(positive_contribution) / positive_count
    covariance += np.cov(negative_contribution) / negative_count
    contrast = np.array([-1.0, 1.0])
    variance = float(contrast @ covariance @ contrast)
    standard_error = math.sqrt(max(variance, 0.0))
    delta = float(auc[1] - auc[0])
    if standard_error > 0:
        z_score = delta / standard_error
    else:
        z_score = 0.0 if delta == 0 else math.copysign(float("inf"), delta)
    p_value = math.erfc(abs(z_score) / math.sqrt(2.0))
    return {
        "baseline_auc": float(auc[0]),
        "candidate_auc": float(auc[1]),
        "delta_auc": delta,
        "standard_error": standard_error,
        "ci95_low": delta - 1.96 * standard_error,
        "ci95_high": delta + 1.96 * standard_error,
        "z_score": z_score,
        "p_value": p_value,
    }


def metric_snapshot(
    labels: np.ndarray, prediction: np.ndarray, users: np.ndarray
) -> dict[str, float]:
    clipped = np.clip(prediction, 1e-07, 1 - 1e-07)
    (gauc, coverage) = grouped_auc(labels, prediction, users)
    return {
        "auc": float(roc_auc_score(labels, prediction)),
        "pr_auc": float(average_precision_score(labels, prediction)),
        "logloss": float(log_loss(labels, clipped, labels=[0, 1])),
        "brier": float(np.mean(np.square(prediction - labels))),
        "ece": expected_calibration_error(labels, prediction),
        "gauc": gauc,
        "gauc_coverage": coverage,
    }


def paired_logloss_interval(
    labels: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
) -> dict[str, float]:
    baseline = np.clip(baseline_prediction, 1e-07, 1 - 1e-07)
    candidate = np.clip(candidate_prediction, 1e-07, 1 - 1e-07)
    baseline_loss = -(labels * np.log(baseline) + (1 - labels) * np.log1p(-baseline))
    candidate_loss = -(labels * np.log(candidate) + (1 - labels) * np.log1p(-candidate))
    difference = candidate_loss - baseline_loss
    delta = float(difference.mean())
    standard_error = float(difference.std(ddof=1) / math.sqrt(len(difference)))
    return {
        "delta_logloss": delta,
        "standard_error": standard_error,
        "ci95_low": delta - 1.96 * standard_error,
        "ci95_high": delta + 1.96 * standard_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.baseline, allow_pickle=False) as baseline_data:
        baseline_label = baseline_data["label"]
        baseline_prediction = baseline_data["prediction"]
        baseline_user = baseline_data["user"]
    with np.load(args.candidate, allow_pickle=False) as candidate_data:
        candidate_label = candidate_data["label"]
        candidate_prediction = candidate_data["prediction"]
        candidate_user = candidate_data["user"]
    if not np.array_equal(baseline_label, candidate_label):
        raise RuntimeError("Prediction files are not label-aligned")
    if not np.array_equal(baseline_user, candidate_user):
        raise RuntimeError("Prediction files are not user-aligned")
    result = {
        "row_count": int(len(baseline_label)),
        "baseline": metric_snapshot(baseline_label, baseline_prediction, baseline_user),
        "candidate": metric_snapshot(
            candidate_label, candidate_prediction, candidate_user
        ),
        "paired_delong": paired_delong(
            baseline_label, baseline_prediction, candidate_prediction
        ),
        "paired_logloss": paired_logloss_interval(
            baseline_label, baseline_prediction, candidate_prediction
        ),
    }
    result["delta"] = {
        name: result["candidate"][name] - result["baseline"][name]
        for name in ("auc", "pr_auc", "logloss", "brier", "ece", "gauc")
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
