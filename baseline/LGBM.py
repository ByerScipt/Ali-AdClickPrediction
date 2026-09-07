from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

TRAIN_DATES = ("2017-05-06", "2017-05-11")
VALID_DATE = "2017-05-12"
TEST_DATE = "2017-05-13"
SEED = 42


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def metric_dict(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int8)
    p = np.clip(np.asarray(predictions, dtype=np.float64), 1e-07, 1.0 - 1e-07)
    return {
        "auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(np.mean((p - y) ** 2)),
        "rows": int(y.size),
        "positive_rate": float(y.mean()),
    }


def split_by_date(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = data["date"].astype(str)
    train = data[(dates >= TRAIN_DATES[0]) & (dates <= TRAIN_DATES[1])].copy()
    valid = data[dates == VALID_DATE].copy()
    test = data[dates == TEST_DATE].copy()
    actual = (len(train), len(valid), len(test))
    if not all(actual):
        raise RuntimeError(f"temporal partition has an empty split: actual={actual}")
    return (train, valid, test)


def run_lgbm(args: argparse.Namespace) -> None:
    from lightgbm import LGBMClassifier, early_stopping

    path = args.data_dir / "sample/DINPointerSample_full_seq2048_btag_time"
    started = time.perf_counter()
    categorical = [
        "hour",
        "weekday",
        "pid",
        "cate_id",
        "cms_group_id",
        "final_gender_code",
        "occupation",
        "brand",
    ]
    behavior = [
        "same_cate_hit_1h",
        "same_cate_cnt_1h",
        "same_brand_hit_1h",
        "same_brand_cnt_1h",
        "same_cate_hit_1d",
        "same_cate_cnt_1d",
        "same_brand_hit_1d",
        "same_brand_cnt_1d",
        "hi_cate_hit_1d",
        "hi_cate_cnt_1d",
        "hi_brand_hit_1d",
        "hi_brand_cnt_1d",
    ]
    numeric = [
        "user_cate_hist_imp",
        "user_cate_hist_clk",
        "user_cate_hist_ctr",
        "user_hist_ctr",
        "user_hist_clk",
        "user_hist_imp",
        "ad_hist_clk",
        "ad_hist_imp",
        "price",
        "age_level",
        "shopping_level",
    ] + behavior
    features = categorical + numeric
    data = pd.read_parquet(path, columns=["date", "clk"] + features)
    (train, valid, test) = split_by_date(data)
    category_dtypes = {
        column: pd.CategoricalDtype(
            categories=pd.Index(pd.unique(data[column].fillna(-1))).sort_values()
        )
        for column in categorical
    }
    for frame in (train, valid, test):
        for column in categorical:
            frame[column] = frame[column].fillna(-1).astype(category_dtypes[column])
        frame.loc[:, numeric] = (
            frame[numeric].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        )
    print(
        f"LGBM categorical features encoded: train={train[features].shape}, valid={valid[features].shape}",
        flush=True,
    )
    model = LGBMClassifier(
        objective="binary",
        random_state=SEED,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=args.cpu_threads,
        verbosity=-1,
    )
    model.fit(
        train[features],
        train["clk"],
        categorical_feature=categorical,
        eval_set=[(valid[features], valid["clk"])],
        eval_metric="auc",
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
    )
    valid_metrics = metric_dict(
        valid["clk"].to_numpy(), model.predict_proba(valid[features])[:, 1]
    )
    test_metrics = metric_dict(
        test["clk"].to_numpy(), model.predict_proba(test[features])[:, 1]
    )
    output = args.output_dir / "metrics.json"
    write_json(
        output,
        {
            "model": "LGBM",
            "protocol": {
                "train": "2017-05-06..2017-05-11",
                "validation": VALID_DATE,
                "test": TEST_DATE,
            },
            "data_path": str(path),
            "feature_set": {"categorical": categorical, "numeric": numeric},
            "config": {
                "seed": SEED,
                "n_estimators_max": 1000,
                "best_iteration": int(model.best_iteration_ or 1000),
                "learning_rate": 0.05,
                "num_leaves": 63,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "cpu_threads": args.cpu_threads,
            },
            "validation": valid_metrics,
            "test": test_metrics,
            "elapsed_seconds": time.perf_counter() - started,
        },
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(args.output_dir / "model.txt"))
    print(
        json.dumps(
            {"model": "LGBM", "validation": valid_metrics, "test": test_metrics},
            ensure_ascii=False,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lgbm"))
    parser.add_argument("--cpu-threads", type=int, default=8)
    run_lgbm(parser.parse_args())


if __name__ == "__main__":
    main()
