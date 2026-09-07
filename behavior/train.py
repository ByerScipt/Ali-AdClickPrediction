from __future__ import annotations

import argparse
import json
import os
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import torch
import torch.distributed as dist
import torch.nn as nn
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from behavior.mixformer import MixFormerCTRBackbone
from behavior.rankmixer import (
    RankMixerBackbone,
    describe_rankmixer_layout,
    semantic_v1_groups,
)


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/mixformer.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--evaluate", choices=["valid", "test"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-head-mixing", action="store_true")
    parser.add_argument("--no-sequence-cross-attention", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    if args.no_head_mixing:
        settings["mixformer_use_head_mixing"] = False
    if args.no_sequence_cross_attention:
        settings["mixformer_use_sequence_cross_attention"] = False
    if args.cpu:
        settings.update(gpu_behavior_store=False, amp=False)
    return (args, settings)


if __name__ == "__main__":
    (OPTIONS, SETTINGS) = parse_options()
else:
    OPTIONS = argparse.Namespace(
        data_dir=Path("data"),
        output_dir=None,
        evaluate=None,
        checkpoint=None,
        resume=False,
        cpu=False,
        config=Path("configs/mixformer.json"),
    )
    SETTINGS = json.loads(
        (Path(__file__).resolve().parents[1] / "configs/mixformer.json").read_text()
    )


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = (OPTIONS.output_dir or Path("outputs") / OPTIONS.config.stem).resolve()


def dedupe_keep_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


DATA_PATH = (
    OPTIONS.data_dir / "sample/DINPointerSample_full_seq2048_btag_time"
).resolve()
BEHAVIOR_STORE_PATH = (
    OPTIONS.data_dir / "processed_data/behavior_store_full_official_btag_time"
).resolve()
MAX_SEQUENCE_LENGTH = SETTINGS.get("max_sequence_length", 1024)
SEED = SETTINGS.get("seed", 42)
EPOCHS = SETTINGS.get("epochs", 8)
EMB_DIM = SETTINGS.get("emb_dim", 8)
EMB_INIT_STD = SETTINGS.get("embedding_init_std", 0.01)
BTAG_EMB_DIM = SETTINGS.get("btag_emb_dim", 4)
TIME_EMB_DIM = SETTINGS.get("time_emb_dim", 8)
ATTN_HIDDEN_DIMS = SETTINGS.get("attn_hidden_dims", [128, 64])
if not ATTN_HIDDEN_DIMS:
    ATTN_HIDDEN_DIMS = [32]
NUM_CROSS_LAYERS = SETTINGS.get("num_cross_layers", 2)
CROSS_RANK = SETTINGS.get("cross_rank", 64)
DEEP_HIDDEN_DIMS = SETTINGS.get("deep_hidden_dims", [128, 64])
if not DEEP_HIDDEN_DIMS:
    DEEP_HIDDEN_DIMS = [64, 32]
FUSION_HIDDEN_DIM = SETTINGS.get("fusion_hidden_dim", 0)
LEARNING_RATE = SETTINGS.get("lr", 0.0005)
WEIGHT_DECAY = SETTINGS.get("weight_decay", 1e-05)
DROPOUT = SETTINGS.get("dropout", 0.1)
GRAD_CLIP = SETTINGS.get("grad_clip", 5.0)
CPU_THREADS = SETTINGS.get("cpu_threads", 12)
GATHER_WORKERS = SETTINGS.get("gather_workers", 1)
PREFETCH_BATCHES = SETTINGS.get("prefetch_batches", 16)
GPU_BEHAVIOR_STORE = SETTINGS.get("gpu_behavior_store", True)
LOG_EVERY_STEPS = SETTINGS.get("log_every_steps", 100)
TRAIN_BATCH_SIZE = SETTINGS.get("train_batch_size", 1024)
EVAL_BATCH_SIZE = SETTINGS.get("eval_batch_size", 1024)
READ_BATCH_SIZE = SETTINGS.get("read_batch_size", 131072)
MAX_TRAIN_STEPS = SETTINGS.get("max_train_steps", None)
MAX_EVAL_STEPS = SETTINGS.get("max_eval_steps", None)
VALIDATE_EVERY_STEPS = SETTINGS.get("validate_every_steps", 1000)
MIN_VALIDATION_CHECKS = SETTINGS.get("min_validation_checks", 10)
PATIENCE_VALIDATION_CHECKS = SETTINGS.get("patience_validation_checks", 5)
AMP_ENABLED = SETTINGS.get("amp", True)
AMP_DTYPE_NAME = SETTINGS.get("amp_dtype", "bf16")
AMP_DTYPE = torch.bfloat16 if AMP_DTYPE_NAME == "bf16" else torch.float16
RESUME = OPTIONS.resume
EARLY_STOPPING_PATIENCE = SETTINGS.get("patience", 3)
MIN_EPOCHS = SETTINGS.get("min_epochs", 4)
MIN_DELTA = SETTINGS.get("min_delta", 1e-05)
USE_DIN = SETTINGS.get("use_din", True)
ATTN_WEIGHT_MODE = SETTINGS.get("attn_weight_mode", "softmax")
INTEREST_EXTRACTOR = SETTINGS.get("interest_extractor", "mixformer")
CROSS_TYPE = SETTINGS.get("cross_type", "mixformer")
RANKMIXER_DIM = SETTINGS.get("rankmixer_dim", 88)
RANKMIXER_LAYERS = SETTINGS.get("rankmixer_layers", 4)
RANKMIXER_EXPANSION = SETTINGS.get("rankmixer_expansion", 2)
MIXFORMER_NUM_HEADS = SETTINGS.get("mixformer_num_heads", 16)
MIXFORMER_HEAD_DIM = SETTINGS.get("mixformer_head_dim", 64)
MIXFORMER_LAYERS = SETTINGS.get("mixformer_layers", 2)
MIXFORMER_ADAPTER_DIM = SETTINGS.get("mixformer_adapter_dim", 16)
MIXFORMER_HEAD_FFN_EXPANSION = SETTINGS.get("mixformer_head_ffn_expansion", 2)
MIXFORMER_SEQUENCE_FFN_EXPANSION = SETTINGS.get("mixformer_sequence_ffn_expansion", 2)
MIXFORMER_TASK_HIDDEN_DIMS = SETTINGS.get("mixformer_task_hidden_dims", [256, 128])
MIXFORMER_GRADIENT_CHECKPOINTING = SETTINGS.get(
    "mixformer_gradient_checkpointing", True
)
MIXFORMER_USE_HEAD_MIXING = SETTINGS.get("mixformer_use_head_mixing", True)
MIXFORMER_USE_SEQUENCE_CROSS_ATTENTION = SETTINGS.get(
    "mixformer_use_sequence_cross_attention", True
)
DISABLE_TQDM = SETTINGS.get("disable_tqdm", True)
EVAL_ONLY = OPTIONS.evaluate is not None
EVAL_SPLIT = OPTIONS.evaluate or "valid"
PREDICTION_PATH = CHECKPOINT_DIR / (EVAL_SPLIT + ".npz") if EVAL_ONLY else None
LAST_CKPT_PATH = CHECKPOINT_DIR / "last.pt"
BEST_CKPT_PATH = (
    OPTIONS.checkpoint.resolve() if OPTIONS.checkpoint else CHECKPOINT_DIR / "best.pt"
)
HISTORY_PATH = CHECKPOINT_DIR / "history.json"
TARGET_COLS = ["cate_id", "brand"]
HIST_COLS = {"cate_id": "hist_cate_seq", "brand": "hist_brand_seq"}
HIST_BTAG_COL = "hist_btag_seq"
HIST_TIME_COL = "hist_time_seq"
BTAG_TO_ID = {"pv": 2, "cart": 3, "fav": 4, "buy": 5}
TIME_GAP_BOUNDARIES = [
    1,
    5,
    10,
    30,
    60,
    300,
    900,
    3600,
    21600,
    86400,
    259200,
    604800,
    1209600,
]
SAME_BEHAVIOR_COLS = [
    "same_cate_hit_1h",
    "same_cate_cnt_1h",
    "same_brand_hit_1h",
    "same_brand_cnt_1h",
    "same_cate_hit_1d",
    "same_cate_cnt_1d",
    "same_brand_hit_1d",
    "same_brand_cnt_1d",
]
HI_BEHAVIOR_COLS = [
    "hi_cate_hit_1d",
    "hi_cate_cnt_1d",
    "hi_brand_hit_1d",
    "hi_brand_cnt_1d",
]
WINDOW6H_BEHAVIOR_COLS = [
    "same_cate_hit_6h",
    "same_cate_cnt_6h",
    "same_brand_hit_6h",
    "same_brand_cnt_6h",
]
LAST_GAP_COLS = ["last_gap_cate", "last_gap_brand"]
BEHAVIOR_DENSE_COLS = [
    *SAME_BEHAVIOR_COLS,
    *HI_BEHAVIOR_COLS,
    *WINDOW6H_BEHAVIOR_COLS,
    *LAST_GAP_COLS,
]
SPARSE_COLS = [
    "pid",
    "cate_id",
    "brand",
    "cms_group_id",
    "cms_segid",
    "final_gender_code",
    "occupation",
    "hour",
    "weekday",
]
DENSE_COLS = [
    "price",
    "age_level",
    "shopping_level",
    "ad_hist_imp",
    "ad_hist_clk",
    "user_hist_imp",
    "user_hist_clk",
    "user_hist_ctr",
    "user_cate_hist_imp",
    "user_cate_hist_clk",
    "user_cate_hist_ctr",
    *BEHAVIOR_DENSE_COLS,
]
REQUIRED_ID_SOURCE_COLS = [*SPARSE_COLS, "age_level", "shopping_level"]
REQUIRED_COLUMNS = dedupe_keep_order(
    [
        "clk",
        "date",
        *(["user"] if EVAL_ONLY else []),
        *REQUIRED_ID_SOURCE_COLS,
        *DENSE_COLS,
        *(["seq_len", "hist_start", "hist_end", "time_stamp"] if USE_DIN else []),
    ]
)
if INTEREST_EXTRACTOR not in {"din", "mixformer"}:
    raise ValueError("interest_extractor must be din or mixformer")
if ATTN_WEIGHT_MODE not in {"raw", "softmax"}:
    raise ValueError("attn_weight_mode must be raw or softmax")
if CROSS_TYPE not in {"low_rank", "rankmixer", "mixformer"}:
    raise ValueError("cross_type must be low_rank, rankmixer, or mixformer")
if CROSS_TYPE == "low_rank" and CROSS_RANK <= 0:
    raise ValueError("cross_rank must be positive")
if CROSS_TYPE == "rankmixer":
    if RANKMIXER_DIM <= 0 or RANKMIXER_LAYERS <= 0 or RANKMIXER_EXPANSION <= 0:
        raise ValueError("RankMixer 要求正的 dim / layers / expansion")
    RANKMIXER_TOKEN_LAYOUT = describe_rankmixer_layout(
        "semantic_v1", SPARSE_COLS, DENSE_COLS
    )
else:
    RANKMIXER_TOKEN_LAYOUT: list[dict[str, object]] = []
if CROSS_TYPE == "mixformer":
    if not USE_DIN:
        raise ValueError(
            "MixFormer 要求启用 joint pointer sequence input（CTRModel_USE_DIN=1）"
        )
    if INTEREST_EXTRACTOR != "mixformer":
        raise ValueError("MixFormer 运行请设置 CTRModel_INTEREST_EXTRACTOR=mixformer")
    if (
        MIXFORMER_NUM_HEADS <= 0
        or MIXFORMER_HEAD_DIM <= 0
        or MIXFORMER_LAYERS <= 0
        or (MIXFORMER_ADAPTER_DIM <= 0)
        or (MIXFORMER_HEAD_FFN_EXPANSION <= 0)
        or (MIXFORMER_SEQUENCE_FFN_EXPANSION <= 0)
        or (not MIXFORMER_TASK_HIDDEN_DIMS)
    ):
        raise ValueError(
            "MixFormer 的 heads / dim / layers / adapter / FFN / task head 必须为正"
        )
    if MIXFORMER_HEAD_DIM % MIXFORMER_NUM_HEADS:
        raise ValueError(
            "CTRModel_MIXFORMER_HEAD_DIM 必须能被 CTRModel_MIXFORMER_NUM_HEADS 整除"
        )
    MIXFORMER_TOKEN_LAYOUT = describe_rankmixer_layout(
        "semantic_v1", SPARSE_COLS, DENSE_COLS
    )
else:
    MIXFORMER_TOKEN_LAYOUT: list[dict[str, object]] = []
if GPU_BEHAVIOR_STORE and (not USE_DIN):
    raise ValueError(
        "CTRModel_GPU_BEHAVIOR_STORE=1 要求启用 DIN pointer sequence storage"
    )
if MAX_SEQUENCE_LENGTH < 0:
    raise ValueError(f"CTRModel_MAX_SEQ_LEN 必须 >= 0，当前为 {MAX_SEQUENCE_LENGTH}")
if GATHER_WORKERS < 1:
    raise ValueError(f"CTRModel_GATHER_WORKERS 必须 >= 1，当前为 {GATHER_WORKERS}")
if (
    VALIDATE_EVERY_STEPS < 0
    or MIN_VALIDATION_CHECKS < 0
    or PATIENCE_VALIDATION_CHECKS < 0
):
    raise ValueError(
        "step-level validation 的 interval / min_checks / patience_checks 不能为负"
    )
if VALIDATE_EVERY_STEPS > 0 and (
    MIN_VALIDATION_CHECKS < 1 or PATIENCE_VALIDATION_CHECKS < 1
):
    raise ValueError(
        "启用 step-level validation 时 min_checks 与 patience_checks 必须为正"
    )
if EVAL_SPLIT not in {"valid", "test"}:
    raise ValueError(f"CTRModel_EVAL_SPLIT 仅支持 valid / test，当前为 {EVAL_SPLIT}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sql_date_list(values: list[date]) -> str:
    return ", ".join((f"DATE '{value.isoformat()}'" for value in values))


def normalize_date(value) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if hasattr(value, "date"):
        return value.date()
    return pd.to_datetime(value).date()


def unwrap_model(model: nn.Module) -> nn.Module:
    while isinstance(model, DDP) or hasattr(model, "_orig_mod"):
        model = model.module if isinstance(model, DDP) else model._orig_mod
    return model


def synchronize_model_buffers(model: nn.Module, src: int = 0) -> None:
    if not dist.is_initialized():
        return
    for buffer in unwrap_model(model).buffers():
        dist.broadcast(buffer, src=src)


def setup_distributed() -> tuple[int, int, int, torch.device]:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL DDP requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        return (rank, world_size, local_rank, torch.device("cuda", local_rank))
    device = torch.device(
        "cuda" if torch.cuda.is_available() and (not OPTIONS.cpu) else "cpu"
    )
    return (rank, world_size, local_rank, device)


def distributed_metadata(data_path: Path, rank: int, world_size: int) -> dict:
    metadata = collect_metadata(data_path) if rank == 0 else None
    if world_size > 1:
        payload = [metadata]
        dist.broadcast_object_list(payload, src=0)
        metadata = payload[0]
    return metadata


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(
        (
            parameter.numel()
            for parameter in module.parameters()
            if parameter.requires_grad
        )
    )


def duckdb_parquet_scan(data_path: Path) -> str:
    if data_path.is_dir():
        return f"read_parquet('{data_path.as_posix()}/**/*.parquet', hive_partitioning=true)"
    return f"read_parquet('{data_path.as_posix()}')"


def load_behavior_store_metadata(store_path: Path) -> dict:
    metadata_path = store_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"找不到 behavior store 元信息: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def collect_metadata(data_path: Path) -> dict:
    con = duckdb.connect()
    parquet_scan = duckdb_parquet_scan(data_path)
    dates_df = con.execute(
        f"""
        SELECT DISTINCT date
        FROM {parquet_scan}
        ORDER BY date
        """
    ).fetchdf()
    dates = [normalize_date(value) for value in dates_df["date"].tolist()]
    if len(dates) <= 2:
        raise RuntimeError(f"可用日期不足，无法切分: dates={dates}, holdout_days={2}")
    train_dates = dates[:-2]
    valid_date = dates[-2]
    test_date = dates[-1]
    int_source_cols = [col for col in REQUIRED_ID_SOURCE_COLS if col != "pid"]
    max_sql_parts = []
    for col in int_source_cols:
        if USE_DIN and False:
            hist_col = HIST_COLS[col]
            max_sql_parts.append(
                f"\n                greatest(\n                    coalesce(max(try_cast({col} AS BIGINT)), -1),\n                    coalesce(max(list_max({hist_col})), -1)\n                ) AS {col}_max\n                ".strip()
            )
        else:
            max_sql_parts.append(
                f"coalesce(max(try_cast({col} AS BIGINT)), -1) AS {col}_max"
            )
    max_sql = ",\n                ".join(max_sql_parts)
    stats_sql_parts = [max_sql] if max_sql else []
    if USE_DIN:
        stats_sql_parts.append("coalesce(max(seq_len), 0) AS max_seq_len")
    stats_sql = ",\n                ".join(stats_sql_parts)
    stats_row = con.execute(
        f"""
        SELECT
            {stats_sql}
        FROM {parquet_scan}
        """
    ).fetchone()
    stat_columns = [f"{col}_max" for col in int_source_cols] + (
        ["max_seq_len"] if USE_DIN else []
    )
    raw_max_stats = dict(zip(stat_columns, stats_row))
    dense_sql = ",\n                ".join(
        [
            part
            for col in DENSE_COLS
            for part in (
                f"avg({col}) AS {col}_mean",
                f"coalesce(stddev_samp({col}), 0) AS {col}_std",
            )
        ]
    )
    train_stats_row = con.execute(
        f"""
        SELECT
            count(*) AS train_rows,
            {dense_sql}
        FROM {parquet_scan}
        WHERE date IN ({sql_date_list(train_dates)})
        """
    ).fetchone()
    train_stats_columns = ["train_rows"] + [
        name for col in DENSE_COLS for name in (f"{col}_mean", f"{col}_std")
    ]
    train_stats = dict(zip(train_stats_columns, train_stats_row))
    valid_rows = con.execute(
        f"""
        SELECT count(*)
        FROM {parquet_scan}
        WHERE date = DATE '{valid_date.isoformat()}'
        """
    ).fetchone()[0]
    test_rows = 0
    if test_date is not None:
        test_rows = con.execute(
            f"""
            SELECT count(*)
            FROM {parquet_scan}
            WHERE date = DATE '{test_date.isoformat()}'
            """
        ).fetchone()[0]
    pid_values: list[str] = []
    if "pid" in REQUIRED_ID_SOURCE_COLS:
        pid_values = (
            con.execute(
                f"""
                SELECT DISTINCT pid
                FROM {parquet_scan}
                WHERE pid IS NOT NULL
                ORDER BY pid
                """
            )
            .fetchdf()["pid"]
            .tolist()
        )
    con.close()
    dense_mean = {}
    dense_std = {}
    for col in DENSE_COLS:
        dense_mean[col] = float(train_stats[f"{col}_mean"])
        std = float(train_stats[f"{col}_std"])
        dense_std[col] = std if std > 0 else 1.0
    source_vocab_sizes: dict[str, int] = {}
    behavior_store_metadata = None
    if USE_DIN:
        behavior_store_metadata = load_behavior_store_metadata(BEHAVIOR_STORE_PATH)
    for col in int_source_cols:
        max_value = int(raw_max_stats[f"{col}_max"] or -1)
        if behavior_store_metadata is not None and col in TARGET_COLS:
            max_value = max(max_value, int(behavior_store_metadata[f"{col}_max"]))
        source_vocab_sizes[col] = max_value + 3
    if "pid" in REQUIRED_ID_SOURCE_COLS:
        source_vocab_sizes["pid"] = len(pid_values) + 2
    model_vocab_sizes = {
        col: source_vocab_sizes[col] for col in SPARSE_COLS if col in source_vocab_sizes
    }
    total_rows = int(train_stats["train_rows"]) + int(valid_rows) + int(test_rows)
    return {
        "dates": dates,
        "train_dates": train_dates,
        "valid_date": valid_date,
        "test_date": test_date,
        "row_counts": {
            "train": int(train_stats["train_rows"]),
            "valid": int(valid_rows),
            "test": int(test_rows),
            "total": total_rows,
        },
        "dense_mean": dense_mean,
        "dense_std": dense_std,
        "source_vocab_sizes": source_vocab_sizes,
        "model_vocab_sizes": model_vocab_sizes,
        "pid_values": pid_values,
        "max_seq_len": min(
            int(raw_max_stats.get("max_seq_len", 0)),
            MAX_SEQUENCE_LENGTH or int(raw_max_stats.get("max_seq_len", 0)),
        ),
        "sequence_storage": "pointer",
        "behavior_store_path": BEHAVIOR_STORE_PATH.as_posix()
        if behavior_store_metadata is not None
        else None,
    }


def split_filter(split_name: str, metadata: dict):
    if split_name == "train":
        return ds.field("date").isin(metadata["train_dates"])
    if split_name == "valid":
        return ds.field("date") == metadata["valid_date"]
    if split_name == "test":
        if metadata["test_date"] is None:
            raise ValueError(
                "当前配置没有测试集，请将 CTRModel_VALID_HOLDOUT_DAYS 设为 >= 2"
            )
        return ds.field("date") == metadata["test_date"]
    raise ValueError(f"未知 split: {split_name}")


def encode_exact_int(values) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.kind == "f":
        arr = np.where(np.isfinite(arr), arr, -1)
    arr = arr.astype(np.int64, copy=False)
    encoded = arr + 2
    encoded[arr < 0] = 1
    return encoded


def encode_pid(values, pid_to_index: dict[str, int]) -> np.ndarray:
    arr = np.asarray(values, dtype=object)
    encoded = np.ones(len(arr), dtype=np.int64)
    if len(arr) == 0:
        return encoded
    mask = pd.isna(arr)
    if (~mask).any():
        mapped = pd.Series(arr[~mask], copy=False).map(pid_to_index)
        encoded[~mask] = mapped.fillna(1).astype(np.int64).to_numpy()
    return encoded


class BehaviorStore:
    FILES = {
        "cate_id": "cate.npy",
        "brand": "brand.npy",
        "btag": "btag.npy",
        "time": "time.npy",
    }

    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.metadata = load_behavior_store_metadata(store_path)
        self.arrays = {
            name: np.load(store_path / filename, mmap_mode="r")
            for (name, filename) in self.FILES.items()
        }
        self.gather_executor = (
            ThreadPoolExecutor(
                max_workers=GATHER_WORKERS, thread_name_prefix="dindcn-gather"
            )
            if GATHER_WORKERS > 1
            else None
        )
        row_count = int(self.metadata["row_count"])
        for name, values in self.arrays.items():
            if len(values) != row_count:
                raise RuntimeError(
                    f"behavior store 长度不一致: field={name}, actual={len(values)}, expected={row_count}"
                )

    def gather(
        self,
        field: str,
        starts: np.ndarray,
        ends: np.ndarray,
        max_seq_len: int,
        *,
        encode_ids: bool = False,
    ) -> np.ndarray:
        return self.gather_many(
            (field,),
            starts,
            ends,
            max_seq_len,
            encode_id_fields={field} if encode_ids else set(),
        )[field]

    def gather_many(
        self,
        fields: tuple[str, ...],
        starts: np.ndarray,
        ends: np.ndarray,
        max_seq_len: int,
        *,
        encode_id_fields: set[str] | frozenset[str] = frozenset(),
    ) -> dict[str, np.ndarray]:
        row_count = len(starts)
        if row_count == 0 or max_seq_len == 0:
            return {
                field: np.zeros((row_count, max_seq_len), dtype=np.int64)
                for field in fields
            }
        lengths = np.clip(ends - starts, 0, max_seq_len)
        positions = np.arange(max_seq_len, dtype=np.int64)[None, :]
        valid = positions < lengths[:, None]
        indices = starts[:, None] + positions
        indices[~valid] = 0

        def gather_field(field: str) -> tuple[str, np.ndarray]:
            output = np.zeros((row_count, max_seq_len), dtype=np.int64)
            values = np.asarray(self.arrays[field][indices], dtype=np.int64)
            if field in encode_id_fields:
                values = values + 2
                values[values < 2] = 1
            output[valid] = values[valid]
            return (field, output)

        if self.gather_executor is None or len(fields) == 1:
            return dict((gather_field(field) for field in fields))
        return dict(self.gather_executor.map(gather_field, fields))


class BatchEncoder:
    def __init__(self, metadata: dict):
        self.dense_mean = metadata["dense_mean"]
        self.dense_std = metadata["dense_std"]
        self.max_seq_len = metadata["max_seq_len"]
        self.source_vocab_sizes = metadata["source_vocab_sizes"]
        self.pid_to_index = {
            value: idx + 2 for (idx, value) in enumerate(metadata["pid_values"])
        }
        self.behavior_store = (
            BehaviorStore(Path(metadata["behavior_store_path"]))
            if USE_DIN
            and metadata["sequence_storage"] == "pointer"
            and (not GPU_BEHAVIOR_STORE)
            else None
        )

    def encode_batch(
        self, record_batch
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        features: dict[str, torch.Tensor] = {}
        encoded_id_cache: dict[str, np.ndarray] = {}
        if EVAL_ONLY:
            group_user = np.asarray(
                record_batch.column("user").to_numpy(zero_copy_only=False),
                dtype=np.int64,
            )
            features["_group_user"] = torch.from_numpy(group_user)
        if USE_DIN:
            hist_end = np.array(
                record_batch.column("hist_end").to_numpy(zero_copy_only=False),
                dtype=np.int64,
                copy=True,
            )
            stored_start = np.array(
                record_batch.column("hist_start").to_numpy(zero_copy_only=False),
                dtype=np.int64,
                copy=True,
            )
            hist_start = np.maximum(stored_start, hist_end - self.max_seq_len)
            seq_len = np.clip(hist_end - hist_start, 0, self.max_seq_len)
            features["seq_len"] = torch.from_numpy(seq_len.astype(np.int64, copy=False))
            if self.behavior_store is not None:
                store_fields = tuple(HIST_COLS)
                store_fields = (*store_fields, "btag")
                store_fields = (*store_fields, "time")
                gathered = self.behavior_store.gather_many(
                    store_fields,
                    hist_start,
                    hist_end,
                    self.max_seq_len,
                    encode_id_fields=frozenset(HIST_COLS),
                )
                for target_col, hist_col in HIST_COLS.items():
                    features[hist_col] = torch.from_numpy(gathered[target_col])
            else:
                features["hist_start"] = torch.from_numpy(
                    hist_start.astype(np.int64, copy=False)
                )
                features["hist_end"] = torch.from_numpy(
                    hist_end.astype(np.int64, copy=False)
                )
            if self.behavior_store is not None:
                btag_seq = gathered["btag"]
                time_seq = gathered["time"]
            if self.behavior_store is not None:
                features[HIST_BTAG_COL] = torch.from_numpy(btag_seq)
                features[HIST_TIME_COL] = torch.from_numpy(time_seq)
            time_stamp = np.array(
                record_batch.column("time_stamp").to_numpy(zero_copy_only=False),
                dtype=np.int64,
                copy=True,
            )
            features["time_stamp"] = torch.from_numpy(time_stamp)
        for col in REQUIRED_ID_SOURCE_COLS:
            values = record_batch.column(col).to_numpy(zero_copy_only=False)
            if col == "pid":
                encoded_id_cache[col] = encode_pid(values, self.pid_to_index)
            else:
                encoded_id_cache[col] = encode_exact_int(values)
        sparse_arrays = []
        for col in SPARSE_COLS:
            encoded = encoded_id_cache[col]
            sparse_arrays.append(encoded)
        sparse_matrix = (
            np.stack(sparse_arrays, axis=1)
            if sparse_arrays
            else np.zeros((0, 0), dtype=np.int64)
        )
        features["sparse"] = torch.from_numpy(
            sparse_matrix.astype(np.int64, copy=False)
        )
        dense_arrays = []
        for col in DENSE_COLS:
            values = np.asarray(
                record_batch.column(col).to_numpy(zero_copy_only=False),
                dtype=np.float32,
            )
            values = np.nan_to_num(values, nan=self.dense_mean[col]).astype(
                np.float32, copy=False
            )
            values = (values - self.dense_mean[col]) / self.dense_std[col]
            dense_arrays.append(values)
        dense_matrix = (
            np.stack(dense_arrays, axis=1)
            if dense_arrays
            else np.zeros((len(sparse_matrix), 0), dtype=np.float32)
        )
        features["dense"] = torch.from_numpy(
            dense_matrix.astype(np.float32, copy=False)
        )
        labels = np.asarray(
            record_batch.column("clk").to_numpy(zero_copy_only=False), dtype=np.float32
        )
        return (features, torch.from_numpy(labels))


class BatchIterator:
    def __init__(
        self,
        data_path: Path,
        metadata: dict,
        encoder: BatchEncoder,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.dataset = ds.dataset(
            data_path.as_posix(), format="parquet", exclude_invalid_files=True
        )
        self.metadata = metadata
        self.encoder = encoder
        self.rank = rank
        self.world_size = world_size

    def iter_batches(
        self,
        split_name: str,
        batch_size: int,
        shuffle: bool,
        max_steps: int | None = None,
        drop_last: bool = False,
    ):
        scanner = self.dataset.scanner(
            columns=REQUIRED_COLUMNS,
            filter=split_filter(split_name, self.metadata),
            batch_size=READ_BATCH_SIZE,
            use_threads=True,
            batch_readahead=16,
            fragment_readahead=8,
        )
        yielded_steps = 0
        pending_features: dict[str, torch.Tensor] | None = None
        pending_labels: torch.Tensor | None = None
        for record_batch_index, record_batch in enumerate(scanner.to_batches()):
            if self.world_size > 1:
                global_rows = record_batch.num_rows
                partition_rank = (self.rank + record_batch_index) % self.world_size
                start = global_rows * partition_rank // self.world_size
                end = global_rows * (partition_rank + 1) // self.world_size
                if end <= start:
                    continue
                record_batch = record_batch.slice(start, end - start)
            (features, labels) = self.encoder.encode_batch(record_batch)
            row_count = labels.size(0)
            if row_count == 0:
                continue
            if pending_labels is not None:
                features = {
                    name: torch.cat([pending_features[name], tensor], dim=0)
                    for (name, tensor) in features.items()
                }
                labels = torch.cat([pending_labels, labels], dim=0)
                row_count = labels.size(0)
                pending_features = None
                pending_labels = None
            if shuffle:
                order = torch.randperm(row_count)
                features = {
                    name: tensor.index_select(0, order)
                    for (name, tensor) in features.items()
                }
                labels = labels.index_select(0, order)
            full_batch_count = row_count // batch_size
            batch_order = (
                torch.randperm(full_batch_count).tolist()
                if shuffle and False
                else range(full_batch_count)
            )
            for batch_idx in batch_order:
                start = int(batch_idx) * batch_size
                end = start + batch_size
                batch_x = {
                    name: tensor[start:end] for (name, tensor) in features.items()
                }
                batch_y = labels[start:end]
                yield (batch_x, batch_y)
                yielded_steps += 1
                if max_steps is not None and yielded_steps >= max_steps:
                    return
            remainder_start = full_batch_count * batch_size
            if remainder_start < row_count:
                pending_features = {
                    name: tensor[remainder_start:]
                    for (name, tensor) in features.items()
                }
                pending_labels = labels[remainder_start:]
        if pending_labels is not None and (not drop_last):
            yield (pending_features, pending_labels)


class Dice(nn.Module):
    def __init__(self, hidden_dim: int, eps: float = 1e-08):
        super().__init__()
        self.bn = nn.BatchNorm1d(hidden_dim, eps=eps)
        self.alpha = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.dim() == 2:
            normalized = self.bn(x)
            alpha = self.alpha
        else:
            normalized = self.bn(x.reshape(-1, original_shape[-1])).reshape(
                original_shape
            )
            alpha = self.alpha.view(*[1] * (x.dim() - 1), -1)
        prob = torch.sigmoid(normalized)
        return prob * x + (1.0 - prob) * alpha * x


class LowRankCrossLayer(nn.Module):
    def __init__(self, input_dim: int, rank: int):
        super().__init__()
        self.reduce = nn.Linear(input_dim, rank, bias=False)
        self.expand = nn.Linear(rank, input_dim, bias=True)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        return x0 * self.expand(self.reduce(xl)) + xl


class CTRModel(nn.Module):
    def __init__(
        self,
        vocab_sizes: dict[str, int],
        sparse_cols: list[str],
        dense_dim: int,
        dense_cols: list[str] | None = None,
        emb_dim: int = 8,
        btag_emb_dim: int = 4,
        time_emb_dim: int = 8,
        attn_hidden_dims: list[int] | tuple[int, ...] = (64,),
        num_cross_layers: int = 2,
        cross_rank: int = 32,
        deep_hidden_dims: list[int] | tuple[int, ...] = (128, 64),
        fusion_hidden_dim: int = 64,
        dropout: float = 0.15,
        use_din: bool = True,
        interest_extractor: str = "din",
        cross_type: str = "low_rank",
        rankmixer_dim: int = 88,
        rankmixer_layers: int = 4,
        rankmixer_expansion: int = 2,
        mixformer_num_heads: int = 16,
        mixformer_head_dim: int = 64,
        mixformer_layers: int = 2,
        mixformer_adapter_dim: int = 16,
        mixformer_head_ffn_expansion: int = 2,
        mixformer_sequence_ffn_expansion: int = 2,
        mixformer_task_hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        mixformer_gradient_checkpointing: bool = True,
        mixformer_use_head_mixing: bool = True,
        mixformer_use_sequence_cross_attention: bool = True,
        attn_weight_mode: str = "softmax",
        embedding_init_std: float = 0.01,
        max_sequence_length: int = 0,
    ):
        super().__init__()
        self.sparse_cols = sparse_cols
        self.use_din = use_din
        self.interest_extractor = interest_extractor
        self.rankmixer_dim = rankmixer_dim
        self.rankmixer_layers = rankmixer_layers
        self.rankmixer_expansion = rankmixer_expansion
        self.use_mixformer = cross_type == "mixformer"
        self.use_din_interest = use_din and (not self.use_mixformer)
        self.mixformer_num_heads = mixformer_num_heads
        self.mixformer_head_dim = mixformer_head_dim
        self.mixformer_layers = mixformer_layers
        self.mixformer_adapter_dim = mixformer_adapter_dim
        self.mixformer_head_ffn_expansion = mixformer_head_ffn_expansion
        self.mixformer_sequence_ffn_expansion = mixformer_sequence_ffn_expansion
        self.mixformer_task_hidden_dims = tuple(mixformer_task_hidden_dims)
        self.mixformer_gradient_checkpointing = mixformer_gradient_checkpointing
        self.mixformer_use_head_mixing = mixformer_use_head_mixing
        self.mixformer_use_sequence_cross_attention = (
            mixformer_use_sequence_cross_attention
        )
        self.attn_weight_mode = attn_weight_mode
        self.input_max_seq_len = max_sequence_length
        self.gpu_behavior_store: dict[str, torch.Tensor] | None = None
        self.target_indices = [sparse_cols.index(col) for col in TARGET_COLS]
        self.non_target_indices = [
            idx for (idx, col) in enumerate(sparse_cols) if col not in TARGET_COLS
        ]
        self.target_name_to_index = {col: sparse_cols.index(col) for col in TARGET_COLS}
        self.sparse_embeddings = nn.ModuleList(
            [
                nn.Embedding(vocab_sizes[col], emb_dim, padding_idx=0)
                for col in sparse_cols
            ]
        )
        self.dropout = nn.Dropout(dropout)
        interest_dim = 0
        self.joint_event_input_dim = 0
        if use_din:
            context_dim = 0
            self.btag_embedding = nn.Embedding(
                max(BTAG_TO_ID.values()) + 1, btag_emb_dim, padding_idx=0
            )
            context_dim += btag_emb_dim
            self.time_embedding = nn.Embedding(
                len(TIME_GAP_BOUNDARIES) + 2, time_emb_dim, padding_idx=0
            )
            self.register_buffer(
                "time_gap_boundaries",
                torch.tensor(TIME_GAP_BOUNDARIES, dtype=torch.long),
            )
            context_dim += time_emb_dim
            joint_id_dim = emb_dim * len(TARGET_COLS)
            self.joint_event_input_dim = joint_id_dim + context_dim
            if self.use_din_interest:
                attention_input_dim = joint_id_dim * 4 + context_dim
                self.event_value_proj = (
                    nn.Linear(self.joint_event_input_dim, joint_id_dim)
                    if context_dim > 0
                    else nn.Identity()
                )
                interest_dim = joint_id_dim
            if self.use_din_interest:
                if attn_weight_mode.startswith("uniform_"):
                    self.attn_mlp = None
                    self.attn_out = None
                else:
                    attn_layers: list[nn.Module] = []
                    current_dim = attention_input_dim
                    for hidden_dim in attn_hidden_dims:
                        attn_layers.append(nn.Linear(current_dim, hidden_dim))
                        attn_layers.append(Dice(hidden_dim))
                        current_dim = hidden_dim
                    self.attn_mlp = nn.Sequential(*attn_layers)
                    self.attn_out = nn.Linear(current_dim, 1)
        self.sequence_encoder = None
        explicit_input_dim = (
            len(self.non_target_indices) + len(self.target_indices)
        ) * emb_dim + dense_dim
        input_dim = explicit_input_dim + interest_dim
        self.input_dim = input_dim
        self.explicit_input_dim = explicit_input_dim
        self.interest_dim = interest_dim
        self.input_norm = nn.Identity()
        if cross_type == "rankmixer":
            if dense_cols is None:
                raise ValueError("semantic_v1 RankMixer requires dense column names")
            (sparse_groups, dense_groups) = semantic_v1_groups(sparse_cols, dense_cols)
            self.rankmixer_semantic_sparse_indices = tuple(
                (
                    tuple((sparse_cols.index(field) for field in group.fields))
                    for group in sparse_groups
                )
            )
            self.rankmixer_semantic_dense_indices = tuple(
                (
                    tuple((dense_cols.index(field) for field in group.fields))
                    for group in dense_groups
                )
            )
            self.rankmixer_semantic_sparse_proj = nn.ModuleList(
                [
                    nn.Linear(len(indices) * emb_dim, rankmixer_dim)
                    for indices in self.rankmixer_semantic_sparse_indices
                ]
            )
            self.rankmixer_semantic_dense_proj = nn.ModuleList(
                [
                    nn.Linear(len(indices), rankmixer_dim)
                    for indices in self.rankmixer_semantic_dense_indices
                ]
            )
            self.rankmixer_token_count = (
                len(sparse_groups) + len(dense_groups) + (1 if use_din else 0)
            )
            if rankmixer_dim % self.rankmixer_token_count:
                raise ValueError(
                    f"rankmixer_dim must be divisible by the number of semantic tokens ({self.rankmixer_token_count})"
                )
            self.rankmixer_interest_proj = (
                nn.Linear(interest_dim, rankmixer_dim) if use_din else None
            )
            self.rankmixer = RankMixerBackbone(
                model_dim=rankmixer_dim,
                token_count=self.rankmixer_token_count,
                num_layers=rankmixer_layers,
                expansion_ratio=rankmixer_expansion,
                dropout=dropout,
            )
            self.cross_layers = nn.ModuleList()
            effective_rank = 0
        elif cross_type == "mixformer":
            if dense_cols is None:
                raise ValueError(
                    "AliCCP MixFormer requires dense column names for semantic_v1 grouping"
                )
            (sparse_groups, dense_groups) = semantic_v1_groups(sparse_cols, dense_cols)
            token_count = len(sparse_groups) + len(dense_groups)
            if token_count != mixformer_num_heads:
                raise ValueError(
                    f"MixFormer head count must equal the semantic_v1 group count for the current AliCCP adapter, got heads={mixformer_num_heads}, groups={token_count}"
                )
            self.mixformer_semantic_sparse_indices = tuple(
                (
                    tuple((sparse_cols.index(field) for field in group.fields))
                    for group in sparse_groups
                )
            )
            self.mixformer_semantic_dense_indices = tuple(
                (
                    tuple((dense_cols.index(field) for field in group.fields))
                    for group in dense_groups
                )
            )
            self.mixformer_semantic_sparse_adapter = nn.ModuleList(
                [
                    nn.Linear(len(indices) * emb_dim, mixformer_adapter_dim)
                    for indices in self.mixformer_semantic_sparse_indices
                ]
            )
            self.mixformer_semantic_dense_adapter = nn.ModuleList(
                [
                    nn.Linear(len(indices), mixformer_adapter_dim)
                    for indices in self.mixformer_semantic_dense_indices
                ]
            )
            self.mixformer = MixFormerCTRBackbone(
                num_heads=mixformer_num_heads,
                head_dim=mixformer_head_dim,
                adapter_dim=mixformer_adapter_dim,
                sequence_input_dim=self.joint_event_input_dim,
                num_layers=mixformer_layers,
                head_ffn_expansion=mixformer_head_ffn_expansion,
                sequence_ffn_expansion=mixformer_sequence_ffn_expansion,
                task_hidden_dims=mixformer_task_hidden_dims,
                dropout=dropout,
                gradient_checkpointing=mixformer_gradient_checkpointing,
                use_head_mixing=mixformer_use_head_mixing,
                use_sequence_cross_attention=mixformer_use_sequence_cross_attention,
            )
            self.cross_layers = nn.ModuleList()
            effective_rank = 0
        else:
            effective_rank = (
                cross_rank
                if cross_type in {"low_rank", "mix"} and cross_rank > 0
                else 0
            )
        if cross_type in {"rankmixer", "mixformer"}:
            pass
        elif cross_type == "low_rank":
            self.cross_layers = nn.ModuleList(
                [
                    LowRankCrossLayer(input_dim, effective_rank)
                    for _ in range(num_cross_layers)
                ]
            )
        else:
            raise ValueError(f"Unsupported cross type: {cross_type}")
        self.cross_rank = effective_rank
        self.cross_type = cross_type
        self.fusion_hidden_dim = fusion_hidden_dim
        if cross_type in {"rankmixer", "mixformer"}:
            self.deep_net = nn.Identity()
            current_dim = (
                rankmixer_dim
                if cross_type == "rankmixer"
                else mixformer_num_heads * mixformer_head_dim
            )
        else:
            deep_layers: list[nn.Module] = []
            current_dim = input_dim
            for hidden_dim in deep_hidden_dims:
                deep_layers.extend(
                    [nn.Linear(current_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
                )
                current_dim = hidden_dim
            self.deep_net = nn.Sequential(*deep_layers)
        if cross_type == "rankmixer":
            self.output_fc = nn.Linear(rankmixer_dim, 1)
        elif cross_type == "mixformer":
            pass
        else:
            self.output_fc = nn.Linear(current_dim, 1)
        self.reset_recommendation_embeddings(embedding_init_std)

    def attach_gpu_behavior_store(self, store_path: Path, device: torch.device) -> None:
        arrays: dict[str, torch.Tensor] = {}
        store_fields = list(HIST_COLS)
        store_fields.append("btag")
        store_fields.append("time")
        for field in store_fields:
            filename = BehaviorStore.FILES[field]
            values = np.load(store_path / filename, mmap_mode="c")
            tensor = torch.from_numpy(values)
            target_dtype = torch.int8 if field == "btag" else torch.int32
            arrays[field] = tensor.to(device=device, dtype=target_dtype)
        self.gpu_behavior_store = arrays

    def materialize_gpu_histories(
        self, batch_x: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if self.gpu_behavior_store is None:
            return batch_x
        materialized = dict(batch_x)
        hist_start = batch_x["hist_start"]
        seq_len = batch_x["seq_len"]
        max_len = self.input_max_seq_len
        positions = torch.arange(
            max_len, device=hist_start.device, dtype=torch.long
        ).unsqueeze(0)
        valid = positions < seq_len.unsqueeze(1)
        indices = hist_start.unsqueeze(1) + positions
        indices = indices.masked_fill(~valid, 0)
        for target_col, hist_col in HIST_COLS.items():
            values = self.gpu_behavior_store[target_col][indices].to(torch.int32) + 2
            values = values.masked_fill(values < 2, 1).masked_fill(~valid, 0)
            materialized[hist_col] = values
        btag = self.gpu_behavior_store["btag"][indices].to(torch.int32)
        materialized[HIST_BTAG_COL] = btag.masked_fill(~valid, 0)
        event_time = self.gpu_behavior_store["time"][indices].to(torch.int32)
        materialized[HIST_TIME_COL] = event_time.masked_fill(~valid, 0)
        return materialized

    def reset_recommendation_embeddings(self, init_std: float) -> None:
        if init_std <= 0:
            raise ValueError(f"embedding_init_std must be positive, got {init_std}")
        embeddings = list(self.sparse_embeddings)
        if self.use_din:
            embeddings.append(self.btag_embedding)
            embeddings.append(self.time_embedding)
        with torch.no_grad():
            for embedding in embeddings:
                nn.init.normal_(embedding.weight, mean=0.0, std=init_std)
                if embedding.padding_idx is not None:
                    embedding.weight[embedding.padding_idx].zero_()

    def build_joint_sequence_event_input(
        self, batch_x: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        seq_len = batch_x["seq_len"]
        max_len = batch_x[HIST_COLS["cate_id"]].size(1)
        mask = torch.arange(max_len, device=seq_len.device).unsqueeze(
            0
        ) < seq_len.unsqueeze(1)
        hist_cate = self.sparse_embeddings[self.target_name_to_index["cate_id"]](
            batch_x[HIST_COLS["cate_id"]]
        )
        hist_brand = self.sparse_embeddings[self.target_name_to_index["brand"]](
            batch_x[HIST_COLS["brand"]]
        )
        context_parts = []
        btag_ids = batch_x[HIST_BTAG_COL].clamp(
            min=0, max=self.btag_embedding.num_embeddings - 1
        )
        context_parts.append(self.btag_embedding(btag_ids))
        gap_seconds = (
            batch_x["time_stamp"].unsqueeze(1) - batch_x[HIST_TIME_COL]
        ).clamp_min(0)
        gap_bucket = torch.bucketize(gap_seconds, self.time_gap_boundaries) + 1
        gap_bucket = gap_bucket.masked_fill(~mask, 0)
        context_parts.append(self.time_embedding(gap_bucket))
        event_parts = [hist_cate, hist_brand]
        event_parts.extend(context_parts)
        return torch.cat(event_parts, dim=-1)

    def activate_joint_interest(
        self, batch_x: dict[str, torch.Tensor], target_embs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        seq_len = batch_x["seq_len"]
        max_len = batch_x[HIST_COLS["cate_id"]].size(1)
        positions = torch.arange(max_len, device=seq_len.device).unsqueeze(0)
        mask = positions < seq_len.unsqueeze(1)
        hist_cate = self.sparse_embeddings[self.target_name_to_index["cate_id"]](
            batch_x[HIST_COLS["cate_id"]]
        )
        hist_brand = self.sparse_embeddings[self.target_name_to_index["brand"]](
            batch_x[HIST_COLS["brand"]]
        )
        hist_id = torch.cat([hist_cate, hist_brand], dim=-1)
        target_id = torch.cat([target_embs[col] for col in TARGET_COLS], dim=-1)
        context_parts = []
        btag_ids = None
        gap_bucket = None
        btag_ids = batch_x[HIST_BTAG_COL].clamp(
            min=0, max=self.btag_embedding.num_embeddings - 1
        )
        context_parts.append(self.btag_embedding(btag_ids))
        gap_seconds = (
            batch_x["time_stamp"].unsqueeze(1) - batch_x[HIST_TIME_COL]
        ).clamp_min(0)
        gap_bucket = torch.bucketize(gap_seconds, self.time_gap_boundaries) + 1
        gap_bucket = gap_bucket.masked_fill(~mask, 0)
        context_parts.append(self.time_embedding(gap_bucket))
        if context_parts:
            event_context = torch.cat(context_parts, dim=-1)
            event_value_input = torch.cat([hist_id, event_context], dim=-1)
        else:
            event_context = None
            event_value_input = hist_id
        score = None
        if self.attn_mlp is not None and self.attn_out is not None:
            expanded_target = target_id.unsqueeze(1).expand(-1, max_len, -1)
            attention_parts = [
                expanded_target,
                hist_id,
                expanded_target - hist_id,
                expanded_target * hist_id,
            ]
            if event_context is not None:
                attention_parts.append(event_context)
            attn_input = torch.cat(attention_parts, dim=-1)
            score = self.attn_out(self.attn_mlp(attn_input)).squeeze(-1)
        event_value = self.event_value_proj(event_value_input)
        return self.pool_interest(score, event_value, mask)

    def forward(self, batch_x: dict[str, torch.Tensor]) -> torch.Tensor:
        batch_x = self.materialize_gpu_histories(batch_x)
        sparse = batch_x["sparse"]
        dense = batch_x["dense"]
        sparse_embs = [
            embedding(sparse[:, index])
            for (index, embedding) in enumerate(self.sparse_embeddings)
        ]
        if self.cross_type == "mixformer":
            semantic_tokens = [
                adapter(torch.cat([sparse_embs[index] for index in indices], dim=1))
                for (adapter, indices) in zip(
                    self.mixformer_semantic_sparse_adapter,
                    self.mixformer_semantic_sparse_indices,
                )
            ]
            semantic_tokens.extend(
                (
                    adapter(dense[:, list(indices)])
                    for (adapter, indices) in zip(
                        self.mixformer_semantic_dense_adapter,
                        self.mixformer_semantic_dense_indices,
                    )
                )
            )
            sequence_events = self.build_joint_sequence_event_input(batch_x)
            return self.mixformer(
                torch.stack(semantic_tokens, dim=1), sequence_events, batch_x["seq_len"]
            )
        explicit_parts: list[torch.Tensor] = []
        if self.non_target_indices:
            explicit_parts.append(
                torch.cat(
                    [sparse_embs[index] for index in self.non_target_indices], dim=1
                )
            )
        target_embs = {
            col: sparse_embs[self.target_name_to_index[col]] for col in TARGET_COLS
        }
        explicit_parts.append(
            torch.cat([target_embs[col] for col in TARGET_COLS], dim=1)
        )
        interest_tensor = None
        if self.use_din_interest:
            interest_tensor = self.activate_joint_interest(batch_x, target_embs)
        x0_parts = list(explicit_parts)
        if interest_tensor is not None:
            x0_parts.append(interest_tensor)
        x0_parts.append(dense)
        x0 = torch.cat(x0_parts, dim=1)
        if self.cross_type == "rankmixer":
            rank_tokens = [
                projector(torch.cat([sparse_embs[index] for index in indices], dim=1))
                for (projector, indices) in zip(
                    self.rankmixer_semantic_sparse_proj,
                    self.rankmixer_semantic_sparse_indices,
                )
            ]
            rank_tokens.extend(
                (
                    projector(dense[:, list(indices)])
                    for (projector, indices) in zip(
                        self.rankmixer_semantic_dense_proj,
                        self.rankmixer_semantic_dense_indices,
                    )
                )
            )
            if interest_tensor is not None:
                assert self.rankmixer_interest_proj is not None
                rank_tokens.append(self.rankmixer_interest_proj(interest_tensor))
            rank_output = self.rankmixer(torch.stack(rank_tokens, dim=1))
            return self.output_fc(rank_output).squeeze(-1)
        x0 = self.input_norm(x0)
        x_cross = x0
        for cross_layer in self.cross_layers:
            x_cross = cross_layer(x0, x_cross)
        x_deep = self.deep_net(x_cross)
        return self.output_fc(x_deep).squeeze(-1)
        x_deep = self.deep_net(x0)
        fusion_input = torch.cat([x_cross, x_deep], dim=1)
        if self.fusion_fc is None:
            return self.output_fc(fusion_input).squeeze(-1)
        fusion_hidden = self.dropout(torch.relu(self.fusion_fc(fusion_input)))
        return self.output_fc(fusion_hidden).squeeze(-1)

    def pool_interest(
        self, score: torch.Tensor | None, value: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if self.attn_weight_mode == "softmax":
            assert score is not None
            score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
            weight = torch.softmax(score, dim=1) * mask
            weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-09)
        elif self.attn_weight_mode == "sigmoid":
            assert score is not None
            weight = torch.sigmoid(score) * mask
        elif self.attn_weight_mode == "raw":
            assert score is not None
            weight = score * mask
        elif self.attn_weight_mode == "uniform_mean":
            weight = mask.to(value.dtype)
            weight = weight / weight.sum(dim=1, keepdim=True).clamp_min(1.0)
        else:
            weight = mask.to(value.dtype)
        return torch.sum(value * weight.unsqueeze(-1), dim=1)


def move_to_device(
    batch_x: dict[str, torch.Tensor], batch_y: torch.Tensor, device: torch.device
):
    batch_x = {
        name: tensor.to(device, non_blocking=True) for (name, tensor) in batch_x.items()
    }
    batch_y = batch_y.to(device, non_blocking=True)
    return (batch_x, batch_y)


def prefetch_batches(iterator, max_prefetch: int):
    if max_prefetch <= 0:
        yield from iterator
        return
    work_queue: queue.Queue = queue.Queue(maxsize=max_prefetch)
    sentinel = object()

    def producer() -> None:
        try:
            for item in iterator:
                work_queue.put(item)
        except BaseException as exc:
            work_queue.put(exc)
        finally:
            work_queue.put(sentinel)

    worker = threading.Thread(
        target=producer, name="dindcn-batch-prefetch", daemon=True
    )
    worker.start()
    while True:
        item = work_queue.get()
        if item is sentinel:
            break
        if isinstance(item, BaseException):
            raise item
        yield item


EVAL_METRIC_KEYS = (
    "loss",
    "auc",
    "logloss",
    "pr_auc",
    "brier",
    "ece",
    "gauc",
    "gauc_coverage",
)


def expected_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, num_bins: int = 20
) -> float:
    bin_index = np.minimum(
        (np.clip(y_pred, 0.0, 1.0) * num_bins).astype(np.int64), num_bins - 1
    )
    counts = np.bincount(bin_index, minlength=num_bins)
    positive_sum = np.bincount(bin_index, weights=y_true, minlength=num_bins)
    prediction_sum = np.bincount(bin_index, weights=y_pred, minlength=num_bins)
    nonempty = counts > 0
    observed = positive_sum[nonempty] / counts[nonempty]
    predicted = prediction_sum[nonempty] / counts[nonempty]
    return float(
        np.sum(counts[nonempty] * np.abs(observed - predicted)) / max(len(y_true), 1)
    )


def grouped_auc(
    y_true: np.ndarray, y_pred: np.ndarray, group_ids: np.ndarray
) -> tuple[float, float]:
    if len(y_true) == 0:
        return (float("nan"), 0.0)
    order = np.lexsort((y_pred, group_ids))
    sorted_group = group_ids[order]
    sorted_pred = y_pred[order]
    sorted_true = y_true[order].astype(np.float64, copy=False)
    group_start = np.empty(len(order), dtype=bool)
    group_start[0] = True
    group_start[1:] = sorted_group[1:] != sorted_group[:-1]
    group_index = np.cumsum(group_start) - 1
    group_starts = np.flatnonzero(group_start)
    local_rank = (
        np.arange(len(order), dtype=np.float64) - group_starts[group_index] + 1.0
    )
    tie_start = group_start.copy()
    tie_start[1:] |= sorted_pred[1:] != sorted_pred[:-1]
    tie_starts = np.flatnonzero(tie_start)
    tie_ends = np.append(tie_starts[1:], len(order))
    average_tie_rank = (local_rank[tie_starts] + local_rank[tie_ends - 1]) * 0.5
    ranks = np.repeat(average_tie_rank, tie_ends - tie_starts)
    group_count = len(group_starts)
    total = np.bincount(group_index, minlength=group_count).astype(np.float64)
    positives = np.bincount(group_index, weights=sorted_true, minlength=group_count)
    negatives = total - positives
    positive_rank_sum = np.bincount(
        group_index, weights=ranks * sorted_true, minlength=group_count
    )
    valid = (positives > 0) & (negatives > 0)
    if not valid.any():
        return (float("nan"), 0.0)
    auc = (
        positive_rank_sum[valid] - positives[valid] * (positives[valid] + 1.0) * 0.5
    ) / (positives[valid] * negatives[valid])
    valid_weight = total[valid]
    return (
        float(np.average(auc, weights=valid_weight)),
        float(valid_weight.sum() / len(y_true)),
    )


def evaluate(
    model: nn.Module,
    batch_iterator: BatchIterator,
    split_name: str,
    batch_size: int,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_steps: int | None = None,
    rank: int = 0,
    world_size: int = 1,
    collect_predictions: bool = False,
) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
    model.eval()
    total_loss = 0.0
    total_rows = 0
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    group_users: list[np.ndarray] = []
    with torch.no_grad():
        eval_batches = prefetch_batches(
            batch_iterator.iter_batches(
                split_name, batch_size, shuffle=False, max_steps=max_steps
            ),
            PREFETCH_BATCHES,
        )
        for batch_x, batch_y in tqdm(
            eval_batches,
            desc=f"{split_name} evaluate",
            leave=False,
            disable=DISABLE_TQDM or rank != 0,
        ):
            group_user = batch_x.pop("_group_user", None)
            (batch_x, batch_y) = move_to_device(batch_x, batch_y, device)
            with torch.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
            ):
                logit = model(batch_x)
                loss = criterion(logit, batch_y)
            batch_size_now = batch_y.size(0)
            total_loss += loss.item() * batch_size_now
            total_rows += batch_size_now
            y_true.append(batch_y.cpu().numpy())
            y_pred.append(torch.sigmoid(logit.float()).cpu().numpy())
            if group_user is not None:
                group_users.append(group_user.numpy())
    local_payload = (
        np.concatenate(y_true) if y_true else np.empty(0, dtype=np.float32),
        np.concatenate(y_pred) if y_pred else np.empty(0, dtype=np.float32),
        np.concatenate(group_users) if group_users else None,
        total_loss,
        total_rows,
    )
    if world_size > 1:
        gathered = [None] * world_size if rank == 0 else None
        dist.gather_object(local_payload, gathered, dst=0)
    else:
        gathered = [local_payload]
    metrics = torch.empty(len(EVAL_METRIC_KEYS), dtype=torch.float64, device=device)
    details = None
    if rank == 0:
        all_true = np.concatenate([item[0] for item in gathered])
        all_pred = np.concatenate([item[1] for item in gathered])
        all_group = None
        if all((item[2] is not None for item in gathered)):
            all_group = np.concatenate([item[2] for item in gathered])
        all_loss = sum((item[3] for item in gathered))
        all_rows = sum((item[4] for item in gathered))
        if all_rows == 0:
            result = (float("nan"),) * len(EVAL_METRIC_KEYS)
        else:
            avg_loss = all_loss / all_rows
            try:
                auc = roc_auc_score(all_true, all_pred)
            except ValueError:
                auc = float("nan")
            try:
                pr_auc = average_precision_score(all_true, all_pred)
            except ValueError:
                pr_auc = float("nan")
            eval_logloss = log_loss(
                all_true, np.clip(all_pred, 1e-07, 1 - 1e-07), labels=[0, 1]
            )
            brier = float(np.mean(np.square(all_pred - all_true)))
            ece = expected_calibration_error(all_true, all_pred)
            (gauc, gauc_coverage) = (
                grouped_auc(all_true, all_pred, all_group)
                if all_group is not None
                else (float("nan"), float("nan"))
            )
            result = (
                avg_loss,
                float(auc),
                float(eval_logloss),
                float(pr_auc),
                brier,
                ece,
                gauc,
                gauc_coverage,
            )
        if collect_predictions:
            details = {"label": all_true, "prediction": all_pred}
            if all_group is not None:
                details["user"] = all_group
        metrics.copy_(torch.tensor(result, dtype=torch.float64, device=device))
    if world_size > 1:
        dist.broadcast(metrics, src=0)
    metric_values = {
        name: float(value)
        for (name, value) in zip(EVAL_METRIC_KEYS, metrics.cpu().tolist())
    }
    return (metric_values, details)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    epoch: int,
    best_val_auc: float,
    best_epoch: int,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "best_val_auc": best_val_auc,
        "best_epoch": best_epoch,
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "metadata": metadata,
        "config": {
            "data_path": DATA_PATH.as_posix(),
            "sequence_storage": "pointer",
            "behavior_store_path": BEHAVIOR_STORE_PATH.as_posix(),
            "max_sequence_length": MAX_SEQUENCE_LENGTH,
            "emb_dim": EMB_DIM,
            "embedding_init_std": EMB_INIT_STD,
            "btag_emb_dim": BTAG_EMB_DIM,
            "time_emb_dim": TIME_EMB_DIM,
            "attn_hidden_dims": ATTN_HIDDEN_DIMS,
            "num_cross_layers": NUM_CROSS_LAYERS,
            "cross_rank": CROSS_RANK,
            "deep_hidden_dims": DEEP_HIDDEN_DIMS,
            "fusion_hidden_dim": FUSION_HIDDEN_DIM,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "read_batch_size": READ_BATCH_SIZE,
            "validate_every_steps": VALIDATE_EVERY_STEPS,
            "min_validation_checks": MIN_VALIDATION_CHECKS,
            "patience_validation_checks": PATIENCE_VALIDATION_CHECKS,
            "gather_workers": GATHER_WORKERS,
            "gpu_behavior_store": GPU_BEHAVIOR_STORE,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "dropout": DROPOUT,
            "include_user": False,
            "include_adgroup": False,
            "use_low_rank": True,
            "use_din": USE_DIN,
            "use_manual_crosses": False,
            "cross_feature_mode": "none",
            "ordinal_as_dense": True,
            "structure": "stacked",
            "attn_use_dice": True,
            "input_norm": False,
            "interest_extractor": INTEREST_EXTRACTOR,
            "interest_mode": "joint",
            "attn_weight_mode": ATTN_WEIGHT_MODE,
            "attn_use_scale": False,
            "din_use_btag": True,
            "din_use_time": True,
            "din_history_control": "normal",
            "include_behavior_seq_len": False,
            "cross_type": CROSS_TYPE,
            "rankmixer_dim": RANKMIXER_DIM,
            "rankmixer_layers": RANKMIXER_LAYERS,
            "rankmixer_expansion": RANKMIXER_EXPANSION,
            "rankmixer_tokenization": "semantic_v1",
            "rankmixer_token_layout": RANKMIXER_TOKEN_LAYOUT,
            "mixformer_num_heads": MIXFORMER_NUM_HEADS,
            "mixformer_head_dim": MIXFORMER_HEAD_DIM,
            "mixformer_layers": MIXFORMER_LAYERS,
            "mixformer_adapter_dim": MIXFORMER_ADAPTER_DIM,
            "mixformer_head_ffn_expansion": MIXFORMER_HEAD_FFN_EXPANSION,
            "mixformer_sequence_ffn_expansion": MIXFORMER_SEQUENCE_FFN_EXPANSION,
            "mixformer_task_hidden_dims": MIXFORMER_TASK_HIDDEN_DIMS,
            "mixformer_tokenization": "semantic_v1",
            "mixformer_token_layout": MIXFORMER_TOKEN_LAYOUT,
            "mixformer_gradient_checkpointing": MIXFORMER_GRADIENT_CHECKPOINTING,
            "mixformer_use_head_mixing": MIXFORMER_USE_HEAD_MIXING,
            "mixformer_use_sequence_cross_attention": MIXFORMER_USE_SEQUENCE_CROSS_ATTENTION,
            "shuffle_mode": "row",
            "behavior_mode": "same_hi_6h_gap",
            "amp_dtype": AMP_DTYPE_NAME,
        },
    }
    torch.save(state, path)


def validate_checkpoint_config(checkpoint: dict, path: Path) -> None:
    runtime = {
        "sequence_storage": "pointer",
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "emb_dim": EMB_DIM,
        "btag_emb_dim": BTAG_EMB_DIM,
        "time_emb_dim": TIME_EMB_DIM,
        "attn_hidden_dims": ATTN_HIDDEN_DIMS,
        "num_cross_layers": NUM_CROSS_LAYERS,
        "cross_rank": CROSS_RANK,
        "deep_hidden_dims": DEEP_HIDDEN_DIMS,
        "fusion_hidden_dim": FUSION_HIDDEN_DIM,
        "dropout": DROPOUT,
        "include_user": False,
        "include_adgroup": False,
        "use_low_rank": True,
        "use_din": USE_DIN,
        "use_manual_crosses": False,
        "cross_feature_mode": "none",
        "ordinal_as_dense": True,
        "structure": "stacked",
        "input_norm": False,
        "cross_type": CROSS_TYPE,
        "rankmixer_dim": RANKMIXER_DIM,
        "rankmixer_layers": RANKMIXER_LAYERS,
        "rankmixer_expansion": RANKMIXER_EXPANSION,
        "rankmixer_tokenization": "semantic_v1",
        "rankmixer_token_layout": RANKMIXER_TOKEN_LAYOUT,
        "mixformer_num_heads": MIXFORMER_NUM_HEADS,
        "mixformer_head_dim": MIXFORMER_HEAD_DIM,
        "mixformer_layers": MIXFORMER_LAYERS,
        "mixformer_adapter_dim": MIXFORMER_ADAPTER_DIM,
        "mixformer_head_ffn_expansion": MIXFORMER_HEAD_FFN_EXPANSION,
        "mixformer_sequence_ffn_expansion": MIXFORMER_SEQUENCE_FFN_EXPANSION,
        "mixformer_task_hidden_dims": MIXFORMER_TASK_HIDDEN_DIMS,
        "mixformer_tokenization": "semantic_v1",
        "mixformer_token_layout": MIXFORMER_TOKEN_LAYOUT,
        "mixformer_gradient_checkpointing": MIXFORMER_GRADIENT_CHECKPOINTING,
        "mixformer_use_head_mixing": MIXFORMER_USE_HEAD_MIXING,
        "mixformer_use_sequence_cross_attention": MIXFORMER_USE_SEQUENCE_CROSS_ATTENTION,
        "behavior_mode": "same_hi_6h_gap",
        "include_behavior_seq_len": False,
    }
    if USE_DIN:
        runtime.update(
            {
                "attn_use_dice": True,
                "interest_extractor": INTEREST_EXTRACTOR,
                "interest_mode": "joint",
                "attn_weight_mode": ATTN_WEIGHT_MODE,
                "attn_use_scale": False,
                "din_use_btag": True,
                "din_use_time": True,
                "din_history_control": "normal",
            }
        )
    legacy_defaults = {
        "din_use_btag": True,
        "din_use_time": True,
        "din_history_control": "normal",
        "include_behavior_seq_len": False,
    }
    saved = checkpoint.get("config", {})
    mismatches = []
    for key, runtime_value in runtime.items():
        if key in saved:
            saved_value = saved[key]
        elif key in legacy_defaults:
            saved_value = legacy_defaults[key]
        else:
            continue
        if saved_value != runtime_value:
            mismatches.append(
                f"{key}: checkpoint={saved_value!r}, runtime={runtime_value!r}"
            )
    if mismatches:
        raise RuntimeError(
            f"checkpoint 配置与当前前向不一致: {path}\n" + "\n".join(mismatches)
        )


def load_checkpoint(
    path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, scaler
) -> tuple[int, float, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    validate_checkpoint_config(checkpoint, path)
    unwrap_model(model).load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    device = next(unwrap_model(model).parameters()).device
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint["best_val_auc"]),
        int(checkpoint["best_epoch"]),
    )


def main() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到 DIN_DCN 数据文件: {DATA_PATH}")
    (rank, world_size, local_rank, device) = setup_distributed()
    is_main_process = rank == 0
    log = print if is_main_process else lambda *args, **kwargs: None
    log("-----环境与配置-----")
    set_seed(SEED + rank)
    torch.set_num_threads(max(1, min(CPU_THREADS, os.cpu_count() or CPU_THREADS)))
    pa.set_cpu_count(max(1, CPU_THREADS))
    pa.set_io_thread_count(max(2, CPU_THREADS // 2))
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    amp_enabled = AMP_ENABLED and device.type == "cuda"
    scaler_enabled = amp_enabled and AMP_DTYPE == torch.float16
    log(f"data_path={DATA_PATH}")
    log(
        f"sequence_storage={'pointer'}, max_sequence_length={MAX_SEQUENCE_LENGTH or 'data_max'}, behavior_store={BEHAVIOR_STORE_PATH}"
    )
    log(
        f"device={device}, rank={rank}, local_rank={local_rank}, world_size={world_size}, cpu_threads={CPU_THREADS}, gather_workers={GATHER_WORKERS}, prefetch_batches={PREFETCH_BATCHES}"
    )
    log(
        f"epochs={EPOCHS}, emb_dim={EMB_DIM}, embedding_init_std={EMB_INIT_STD}, attn_hidden_dims={ATTN_HIDDEN_DIMS}, num_cross_layers={NUM_CROSS_LAYERS}, cross_rank={CROSS_RANK}, deep_hidden_dims={DEEP_HIDDEN_DIMS}, fusion_hidden_dim={FUSION_HIDDEN_DIM}"
    )
    log(
        f"train_batch_size={TRAIN_BATCH_SIZE}, eval_batch_size={EVAL_BATCH_SIZE}, read_batch_size={READ_BATCH_SIZE}, grad_clip={GRAD_CLIP}, validate_every_steps={VALIDATE_EVERY_STEPS}, min_validation_checks={MIN_VALIDATION_CHECKS}, patience_validation_checks={PATIENCE_VALIDATION_CHECKS}"
    )
    log(json.dumps(SETTINGS, indent=2))
    log("-----数据元信息统计-----")
    metadata = distributed_metadata(DATA_PATH, rank, world_size)
    log(
        f"时间范围:{metadata['dates'][0]}~{metadata['dates'][-1]} | 训练集={metadata['row_counts']['train']} | 验证集={metadata['row_counts']['valid']} | 测试集={metadata['row_counts']['test']} | max_seq_len={metadata['max_seq_len']}"
    )
    log(f"训练日期: {metadata['train_dates'][0]}~{metadata['train_dates'][-1]}")
    log(f"验证日期: {metadata['valid_date']}, 测试日期: {metadata['test_date']}")
    log(
        f"pid_cardinality={len(metadata['pid_values'])}, sparse_cols={SPARSE_COLS}, dense_cols={DENSE_COLS}\n"
    )
    encoder = BatchEncoder(metadata)
    batch_iterator = BatchIterator(
        DATA_PATH, metadata, encoder, rank=rank, world_size=world_size
    )
    log("-----开始构建模型-----")
    model = CTRModel(
        vocab_sizes=metadata["model_vocab_sizes"],
        sparse_cols=SPARSE_COLS,
        dense_dim=len(DENSE_COLS),
        dense_cols=DENSE_COLS,
        emb_dim=EMB_DIM,
        btag_emb_dim=BTAG_EMB_DIM,
        time_emb_dim=TIME_EMB_DIM,
        attn_hidden_dims=ATTN_HIDDEN_DIMS,
        num_cross_layers=NUM_CROSS_LAYERS,
        cross_rank=CROSS_RANK,
        deep_hidden_dims=DEEP_HIDDEN_DIMS,
        fusion_hidden_dim=FUSION_HIDDEN_DIM,
        dropout=DROPOUT,
        use_din=USE_DIN,
        interest_extractor=INTEREST_EXTRACTOR,
        cross_type=CROSS_TYPE,
        rankmixer_dim=RANKMIXER_DIM,
        rankmixer_layers=RANKMIXER_LAYERS,
        rankmixer_expansion=RANKMIXER_EXPANSION,
        mixformer_num_heads=MIXFORMER_NUM_HEADS,
        mixformer_head_dim=MIXFORMER_HEAD_DIM,
        mixformer_layers=MIXFORMER_LAYERS,
        mixformer_adapter_dim=MIXFORMER_ADAPTER_DIM,
        mixformer_head_ffn_expansion=MIXFORMER_HEAD_FFN_EXPANSION,
        mixformer_sequence_ffn_expansion=MIXFORMER_SEQUENCE_FFN_EXPANSION,
        mixformer_task_hidden_dims=MIXFORMER_TASK_HIDDEN_DIMS,
        mixformer_gradient_checkpointing=MIXFORMER_GRADIENT_CHECKPOINTING,
        mixformer_use_head_mixing=MIXFORMER_USE_HEAD_MIXING,
        mixformer_use_sequence_cross_attention=MIXFORMER_USE_SEQUENCE_CROSS_ATTENTION,
        attn_weight_mode=ATTN_WEIGHT_MODE,
        embedding_init_std=EMB_INIT_STD,
        max_sequence_length=metadata["max_seq_len"],
    ).to(device)
    if GPU_BEHAVIOR_STORE:
        store_load_started = time.perf_counter()
        model.attach_gpu_behavior_store(BEHAVIOR_STORE_PATH, device)
        if world_size > 1:
            dist.barrier()
        log(
            f"GPU behavior store 加载完成: {time.perf_counter() - store_load_started:.2f}s"
        )
    log(f"trainable_params={count_trainable_parameters(model):,}")
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=True,
            gradient_as_bucket_view=True,
        )
    criterion = nn.BCEWithLogitsLoss()
    if EVAL_ONLY:
        if not BEST_CKPT_PATH.exists():
            raise FileNotFoundError(f"找不到待评估 checkpoint: {BEST_CKPT_PATH}")
        checkpoint = torch.load(BEST_CKPT_PATH, map_location="cpu", weights_only=False)
        validate_checkpoint_config(checkpoint, BEST_CKPT_PATH)
        unwrap_model(model).load_state_dict(checkpoint["model_state"])
        synchronize_model_buffers(model)
        evaluation_model = model.module if isinstance(model, DDP) else model
        eval_started = time.perf_counter()
        (eval_metrics, predictions) = evaluate(
            evaluation_model,
            batch_iterator,
            EVAL_SPLIT,
            EVAL_BATCH_SIZE,
            criterion,
            device,
            amp_enabled,
            AMP_DTYPE,
            max_steps=MAX_EVAL_STEPS,
            rank=rank,
            world_size=world_size,
            collect_predictions=True,
        )
        log(
            f"{EVAL_SPLIT} checkpoint 评估 | "
            + " | ".join(
                (f"{name}={value:.8f}" for (name, value) in eval_metrics.items())
            )
            + f" | elapsed={time.perf_counter() - eval_started:.2f}s"
        )
        if (
            is_main_process
            and PREDICTION_PATH is not None
            and (predictions is not None)
        ):
            PREDICTION_PATH.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(PREDICTION_PATH, **predictions)
            metrics_path = PREDICTION_PATH.with_suffix(".metrics.json")
            metrics_path.write_text(
                json.dumps(eval_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            log(f"预测与扩展指标已保存: {PREDICTION_PATH}, {metrics_path}")
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        return
    optimizer_kwargs = {"lr": LEARNING_RATE, "weight_decay": WEIGHT_DECAY}
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.Adam(model.parameters(), **optimizer_kwargs)
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(device=device.type, enabled=scaler_enabled)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
    start_epoch = 1
    best_val_auc = float("-inf")
    best_epoch = 0
    if RESUME and LAST_CKPT_PATH.exists():
        (start_epoch, best_val_auc, best_epoch) = load_checkpoint(
            LAST_CKPT_PATH, model, optimizer, scaler
        )
        log(
            f"从 checkpoint 恢复训练: {LAST_CKPT_PATH} | next_epoch={start_epoch} | best_val_auc={best_val_auc:.4f}"
        )
    history = []
    if is_main_process and RESUME and HISTORY_PATH.exists():
        try:
            history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            history = []
    log("\n-----开始训练模型-----")
    train_step_limit = MAX_TRAIN_STEPS
    if train_step_limit is None and world_size > 1:
        train_step_limit = metadata["row_counts"]["train"] // (
            TRAIN_BATCH_SIZE * world_size
        )
    step_validation_enabled = VALIDATE_EVERY_STEPS > 0
    validation_check_count = max(start_epoch - 1, 0) if step_validation_enabled else 0
    stale_validation_checks = 0
    global_step = (start_epoch - 1) * train_step_limit

    def run_step_validation(
        epoch: int,
        step_index: int,
        global_step_index: int,
        epoch_start: float,
        local_loss_sum: float,
        local_rows: int,
    ) -> bool:
        nonlocal \
            best_val_auc, \
            best_epoch, \
            validation_check_count, \
            stale_validation_checks
        train_stats = torch.tensor(
            [local_loss_sum, local_rows], dtype=torch.float64, device=device
        )
        if world_size > 1:
            dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
        rows_so_far = int(train_stats[1].item())
        train_loss_so_far = float(train_stats[0].item() / max(rows_so_far, 1))
        synchronize_model_buffers(model)
        evaluation_model = model.module if isinstance(model, DDP) else model
        validation_start = time.perf_counter()
        (val_metrics, _) = evaluate(
            evaluation_model,
            batch_iterator,
            "valid",
            EVAL_BATCH_SIZE,
            criterion,
            device,
            amp_enabled,
            AMP_DTYPE,
            max_steps=MAX_EVAL_STEPS,
            rank=rank,
            world_size=world_size,
        )
        validation_seconds = time.perf_counter() - validation_start
        validation_check_count += 1
        val_auc = val_metrics["auc"]
        improved = val_auc > best_val_auc + MIN_DELTA
        if improved:
            best_val_auc = val_auc
            best_epoch = epoch
            stale_validation_checks = 0
        else:
            stale_validation_checks += 1
        elapsed = time.perf_counter() - epoch_start
        log(
            f"step验证#{validation_check_count} | epoch={epoch} step={step_index}/{train_step_limit or '?'} global_step={global_step_index} | train_loss_so_far={train_loss_so_far:.4f} | val_auc={val_auc:.8f} | val_logloss={val_metrics['logloss']:.8f} | validation={validation_seconds:.2f}s | stale_checks={stale_validation_checks} | elapsed={elapsed:.2f}s"
        )
        if is_main_process:
            validation_record = {
                "event": "step_validation",
                "epoch": epoch,
                "step_in_epoch": step_index,
                "global_step": global_step_index,
                "validation_check": validation_check_count,
                "train_loss_so_far": train_loss_so_far,
                "train_rows_so_far": rows_so_far,
                "val_loss": val_metrics["loss"],
                "val_auc": val_auc,
                "val_logloss": val_metrics["logloss"],
                "val_pr_auc": val_metrics["pr_auc"],
                "val_brier": val_metrics["brier"],
                "val_ece": val_metrics["ece"],
                "validation_seconds": validation_seconds,
                "stale_validation_checks": stale_validation_checks,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            history.append(validation_record)
            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            HISTORY_PATH.write_text(
                json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if improved:
                save_checkpoint(
                    BEST_CKPT_PATH,
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    best_val_auc,
                    best_epoch,
                    metadata,
                )
        should_stop = (
            validation_check_count >= MIN_VALIDATION_CHECKS
            and stale_validation_checks >= PATIENCE_VALIDATION_CHECKS
        )
        if should_stop:
            log(
                f"step-level early stop: 连续{PATIENCE_VALIDATION_CHECKS}次完整验证未提升，最佳AUC={best_val_auc:.8f}（epoch={best_epoch}）"
            )
        model.train()
        return should_stop

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_rows = 0
        stop_after_validation = False
        last_step_validation_index = 0
        step_index = 0
        train_batches = batch_iterator.iter_batches(
            "train",
            TRAIN_BATCH_SIZE,
            shuffle=True,
            max_steps=train_step_limit,
            drop_last=world_size > 1,
        )
        pbar = tqdm(
            prefetch_batches(train_batches, PREFETCH_BATCHES),
            desc=f"第{epoch}/{EPOCHS}轮",
            leave=False,
            disable=DISABLE_TQDM or not is_main_process,
        )
        for step_index, (batch_x, batch_y) in enumerate(pbar, start=1):
            (batch_x, batch_y) = move_to_device(batch_x, batch_y, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=AMP_DTYPE, enabled=amp_enabled
            ):
                logit = model(batch_x)
                loss = criterion(logit, batch_y)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if GRAD_CLIP > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            batch_size_now = batch_y.size(0)
            train_loss_sum += loss.item() * batch_size_now
            train_rows += batch_size_now
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if (
                is_main_process
                and LOG_EVERY_STEPS > 0
                and (step_index % LOG_EVERY_STEPS == 0)
            ):
                elapsed = time.perf_counter() - epoch_start
                log(
                    f"epoch={epoch} step={step_index}/{train_step_limit or '?'} | loss={loss.item():.4f} | local_throughput={train_rows / max(elapsed, 1e-09):,.0f} samples/s"
                )
            if step_validation_enabled:
                at_interval = step_index % VALIDATE_EVERY_STEPS == 0
                at_epoch_end = (
                    train_step_limit is not None and step_index >= train_step_limit
                )
                if at_interval or at_epoch_end:
                    last_step_validation_index = step_index
                    stop_after_validation = run_step_validation(
                        epoch,
                        step_index,
                        global_step,
                        epoch_start,
                        train_loss_sum,
                        train_rows,
                    )
                    if stop_after_validation:
                        break
        if (
            step_validation_enabled
            and (not stop_after_validation)
            and (train_rows > 0)
            and (last_step_validation_index != step_index)
        ):
            stop_after_validation = run_step_validation(
                epoch, step_index, global_step, epoch_start, train_loss_sum, train_rows
            )
        if step_validation_enabled:
            train_seconds = time.perf_counter() - epoch_start
            train_stats = torch.tensor(
                [train_loss_sum, train_rows], dtype=torch.float64, device=device
            )
            if world_size > 1:
                dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
            global_train_rows = int(train_stats[1].item())
            train_loss = float(train_stats[0].item() / max(global_train_rows, 1))
            if is_main_process and (not stop_after_validation):
                save_checkpoint(
                    LAST_CKPT_PATH,
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    best_val_auc,
                    best_epoch,
                    metadata,
                )
            log(
                f"第{epoch}/{EPOCHS}轮 step-validation训练结束 | train_loss={train_loss:.4f} | train={train_seconds:.2f}s | checks={validation_check_count} | best_auc={best_val_auc:.8f}"
            )
            if stop_after_validation:
                break
            continue
        train_seconds = time.perf_counter() - epoch_start
        train_stats = torch.tensor(
            [train_loss_sum, train_rows], dtype=torch.float64, device=device
        )
        if world_size > 1:
            dist.all_reduce(train_stats, op=dist.ReduceOp.SUM)
        global_train_rows = int(train_stats[1].item())
        train_loss = float(train_stats[0].item() / max(global_train_rows, 1))
        synchronize_model_buffers(model)
        evaluation_model = model.module if isinstance(model, DDP) else model
        eval_start = time.perf_counter()
        (val_metrics, _) = evaluate(
            evaluation_model,
            batch_iterator,
            "valid",
            EVAL_BATCH_SIZE,
            criterion,
            device,
            amp_enabled,
            AMP_DTYPE,
            max_steps=MAX_EVAL_STEPS,
            rank=rank,
            world_size=world_size,
        )
        val_loss = val_metrics["loss"]
        val_auc = val_metrics["auc"]
        val_logloss = val_metrics["logloss"]
        eval_seconds = time.perf_counter() - eval_start
        epoch_seconds = train_seconds + eval_seconds
        train_samples_per_second = global_train_rows / max(train_seconds, 1e-09)
        log(
            f"第{epoch}/{EPOCHS}轮 | 训练集损失={train_loss:.4f} | 验证集损失={val_loss:.4f} | 验证集auc={val_auc:.4f} | 验证集logloss={val_logloss:.4f} | 训练={train_seconds:.2f}s | 验证={eval_seconds:.2f}s | 总计={epoch_seconds:.2f}s | 训练吞吐={train_samples_per_second:,.0f} samples/s"
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_auc": val_auc,
            "val_logloss": val_logloss,
            "val_pr_auc": val_metrics["pr_auc"],
            "val_brier": val_metrics["brier"],
            "val_ece": val_metrics["ece"],
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
            "epoch_seconds": epoch_seconds,
            "train_rows": global_train_rows,
            "train_samples_per_second": train_samples_per_second,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        improved = val_auc > best_val_auc + MIN_DELTA
        if improved:
            best_val_auc = val_auc
            best_epoch = epoch
        if is_main_process:
            history.append(epoch_record)
            HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            HISTORY_PATH.write_text(
                json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            save_checkpoint(
                LAST_CKPT_PATH,
                model,
                optimizer,
                scaler,
                epoch,
                best_val_auc,
                best_epoch,
                metadata,
            )
            if improved:
                save_checkpoint(
                    BEST_CKPT_PATH,
                    model,
                    optimizer,
                    scaler,
                    epoch,
                    best_val_auc,
                    best_epoch,
                    metadata,
                )
        if (
            not improved
            and epoch >= MIN_EPOCHS
            and (epoch - best_epoch >= EARLY_STOPPING_PATIENCE)
        ):
            log(
                f"验证AUC连续{EARLY_STOPPING_PATIENCE}轮未提升，提前停止训练。最佳AUC={best_val_auc:.4f}, 最佳轮次={best_epoch}"
            )
            break
    log(
        f"训练结束，最佳AUC={best_val_auc:.4f}, 最佳轮次={best_epoch}, best_ckpt={BEST_CKPT_PATH}"
    )
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
