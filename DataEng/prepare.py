import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
RAW_DIR = DATA_DIR / "raw_data"
PROCESSED_DIR = DATA_DIR / "processed_data"
SAMPLE_DIR = DATA_DIR / "sample"
RAW_SAMPLE_PATH = RAW_DIR / "raw_sample.csv"
AD_PATH = RAW_DIR / "ad_feature.csv"
USER_PATH = RAW_DIR / "user_profile.csv"
BEHAVIOR_PATH = PROCESSED_DIR / "behavior_log.parquet"
OFFICIAL_START_TS = 1492790400
OFFICIAL_END_TS = 1494691199
BEHAVIOR_START_TS = 1492790400
BEHAVIOR_END_TS = 1494691199
SAMPLE_BATCH_SIZE = 500000
BEHAVIOR_BATCH_SIZE = 1000000
WRITE_BUFFER_ROWS = 200000
ROW_GROUP_SIZE = 200000
DUCKDB_THREADS = 8
DUCKDB_MEMORY_LIMIT = "8GB"
PARQUET_COMPRESSION = "zstd"
FORCE_REBUILD_STATIC = False
FORCE_REBUILD_BEHAVIOR = False


def ensure_directories() -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / "duckdb_tmp"
    path.mkdir(exist_ok=True)
    return path


def build_paths() -> tuple[Path, Path]:
    return (
        PROCESSED_DIR / "DINStaticSample.parquet",
        PROCESSED_DIR / "behavior_log_din_sorted_btag_time_official.parquet",
    )


def connect_duckdb(temp_dir: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET TimeZone='Asia/Shanghai'")
    con.execute(f"PRAGMA threads={DUCKDB_THREADS}")
    con.execute(f"PRAGMA memory_limit='{DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"PRAGMA temp_directory='{temp_dir.as_posix()}'")
    return con


def sample_scan_sql() -> str:
    return f"read_csv_auto('{RAW_SAMPLE_PATH.as_posix()}', header=true, nullstr='NULL')"


def build_static_sample(static_sample_path: Path, temp_dir: Path) -> None:
    if static_sample_path.exists() and (not FORCE_REBUILD_STATIC):
        print(f"复用静态特征缓存: {static_sample_path}")
        return
    if static_sample_path.exists():
        static_sample_path.unlink()
    print("-------读取并处理静态特征-------")
    print("使用 DuckDB 生成静态特征并落盘缓存...")
    shifted_ts_sql = f"(to_timestamp(s.time_stamp::BIGINT) + INTERVAL '{0} hour')"
    select_sql = f"""
        WITH sample_base AS (
            SELECT row_number() OVER () AS sample_id, *
            FROM {sample_scan_sql()}
        ),
        joined AS (
            SELECT
                s.sample_id,
                s.time_stamp::BIGINT AS time_stamp,
                s.clk::INTEGER AS clk,
                s.pid::VARCHAR AS pid,
                s.user::INTEGER AS user,
                s.adgroup_id::INTEGER AS adgroup_id,
                coalesce(a.cate_id::INTEGER, -1) AS cate_id,
                coalesce(u.cms_group_id::INTEGER, -1) AS cms_group_id,
                coalesce(u.cms_segid::INTEGER, -1) AS cms_segid,
                coalesce(u.final_gender_code::INTEGER, -1) AS final_gender_code,
                coalesce(u.occupation::INTEGER, -1) AS occupation,
                coalesce(u.age_level::INTEGER, -1) AS age_level,
                coalesce(u.shopping_level::INTEGER, -1) AS shopping_level,
                coalesce(a.brand::INTEGER, -1) AS brand,
                coalesce(a.price::DOUBLE, -1.0) AS price,
                EXTRACT(HOUR FROM {shifted_ts_sql})::INTEGER AS hour,
                ((EXTRACT(ISODOW FROM {shifted_ts_sql})::INTEGER + 6) % 7) AS weekday,
                CAST({shifted_ts_sql} AS DATE) AS date,
                row_number() OVER (
                    PARTITION BY s.adgroup_id
                    ORDER BY s.time_stamp::BIGINT, s.sample_id
                ) - 1 AS ad_hist_imp,
                coalesce(sum(s.clk::INTEGER) OVER (
                    PARTITION BY s.adgroup_id
                    ORDER BY s.time_stamp::BIGINT, s.sample_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS ad_hist_clk,
                row_number() OVER (
                    PARTITION BY s.user
                    ORDER BY s.time_stamp::BIGINT, s.sample_id
                ) - 1 AS user_hist_imp,
                coalesce(sum(s.clk::INTEGER) OVER (
                    PARTITION BY s.user
                    ORDER BY s.time_stamp::BIGINT, s.sample_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS user_hist_clk,
                row_number() OVER (
                    PARTITION BY s.user, coalesce(a.cate_id::INTEGER, -1)
                    ORDER BY s.time_stamp::BIGINT, s.sample_id
                ) - 1 AS user_cate_hist_imp,
                coalesce(sum(s.clk::INTEGER) OVER (
                    PARTITION BY s.user, coalesce(a.cate_id::INTEGER, -1)
                    ORDER BY s.time_stamp::BIGINT, s.sample_id
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ), 0) AS user_cate_hist_clk,
                concat(
                    coalesce(u.final_gender_code::INTEGER, -1)::VARCHAR,
                    '_',
                    coalesce(a.cate_id::INTEGER, -1)::VARCHAR
                ) AS gender_cate_cross
            FROM sample_base AS s
            LEFT JOIN read_csv_auto('{AD_PATH.as_posix()}', header=true, nullstr='NULL') AS a
            USING (adgroup_id)
            LEFT JOIN read_csv_auto('{USER_PATH.as_posix()}', header=true, nullstr='NULL') AS u
            ON s.user = u.userid
        )
        SELECT
            time_stamp,
            clk,
            pid,
            user,
            adgroup_id,
            cate_id,
            cms_group_id,
            cms_segid,
            final_gender_code,
            occupation,
            age_level,
            shopping_level,
            brand,
            price,
            hour,
            weekday,
            date,
            ad_hist_imp,
            ad_hist_clk,
            user_hist_imp,
            user_hist_clk,
            (user_hist_clk + 0.25) / (user_hist_imp + 5) AS user_hist_ctr,
            user_cate_hist_imp,
            user_cate_hist_clk,
            (user_cate_hist_clk + 0.25) / (user_cate_hist_imp + 5) AS user_cate_hist_ctr,
            gender_cate_cross
        FROM joined
        ORDER BY user, time_stamp, sample_id
    """
    con = connect_duckdb(temp_dir)
    con.execute(
        f"""
        COPY ({select_sql}) TO '{static_sample_path.as_posix()}' (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE {ROW_GROUP_SIZE}
        )
        """
    )
    (row_count, user_count) = con.execute(
        f"""
        SELECT count(*) AS row_count, count(DISTINCT user) AS user_count
        FROM read_parquet('{static_sample_path.as_posix()}')
        """
    ).fetchone()
    con.close()
    print("静态特征处理完成！")
    print(f"处理后的 sample 形状({row_count}, 26)")
    print(f"静态特征缓存已保存到: {static_sample_path}\n")
    print(f"静态特征中的用户数: {user_count}\n")


def build_behavior_cache(
    behavior_cache_path: Path, static_sample_path: Path, temp_dir: Path
) -> None:
    if behavior_cache_path.exists() and (not FORCE_REBUILD_BEHAVIOR):
        print(f"复用排序后的行为日志缓存: {behavior_cache_path}\n")
        return
    if behavior_cache_path.exists():
        behavior_cache_path.unlink()
    print("-------读取并处理行为日志-------")
    print("使用 DuckDB 过滤并按 user + time_stamp 排序行为日志...")
    selected_cols = [
        "b.user::INTEGER AS user",
        "b.time_stamp::BIGINT AS time_stamp",
        "coalesce(b.cate::INTEGER, -1) AS cate",
        "coalesce(b.brand::INTEGER, -1) AS brand",
    ]
    selected_cols.append(
        "CASE lower(coalesce(b.btag::VARCHAR, '')) WHEN 'pv' THEN 2 WHEN 'cart' THEN 3 WHEN 'fav' THEN 4 WHEN 'buy' THEN 5 ELSE 1 END::TINYINT AS btag"
    )
    selected_cols.append("b.time_stamp::BIGINT AS hist_time_stamp")
    filters = [
        f"b.time_stamp::BIGINT BETWEEN {BEHAVIOR_START_TS} AND {BEHAVIOR_END_TS}"
    ]
    where_sql = "WHERE " + " AND ".join(filters)
    con = connect_duckdb(temp_dir)
    con.execute(
        f"""
        COPY (
            SELECT {", ".join(selected_cols)}
            FROM read_parquet('{BEHAVIOR_PATH.as_posix()}') AS b
            INNER JOIN (
                SELECT DISTINCT user
                FROM read_parquet('{static_sample_path.as_posix()}')
            ) AS s USING (user)
            {where_sql}
            ORDER BY b.user, b.time_stamp
        ) TO '{behavior_cache_path.as_posix()}' (
            FORMAT PARQUET,
            COMPRESSION ZSTD,
            ROW_GROUP_SIZE {max(ROW_GROUP_SIZE, 500000)}
        )
        """
    )
    (row_count, user_count) = con.execute(
        f"""
        SELECT count(*) AS row_count, count(DISTINCT user) AS user_count
        FROM read_parquet('{behavior_cache_path.as_posix()}')
        """
    ).fetchone()
    con.close()
    print(f"过滤后的行为日志已保存到: {behavior_cache_path}")
    print(f"行为日志行数: {row_count}, 命中用户数: {user_count}\n")


NUM_SHARDS = 8
POINTER_WORKERS = 8
FORCE_REBUILD_POINTER = False
STORE_BATCH_SIZE = 2000000


def pointer_paths() -> tuple[Path, Path, Path]:
    return (
        PROCESSED_DIR / "behavior_store_full_official_btag_time",
        PROCESSED_DIR / f"DINStaticShards_{NUM_SHARDS}",
        SAMPLE_DIR / "DINPointerSample_full_seq2048_btag_time",
    )


def remove_generated_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def build_behavior_store(
    behavior_cache_path: Path, static_sample_path: Path, store_dir: Path, temp_dir: Path
) -> dict:
    metadata_path = store_dir / "metadata.json"
    if metadata_path.exists() and (not FORCE_REBUILD_POINTER):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(f"Reusing behavior store: {store_dir}")
        return metadata
    tmp_dir = store_dir.with_name(f"{store_dir.name}.tmp")
    remove_generated_path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    con = connect_duckdb(temp_dir)
    (row_count, max_behavior_user, cate_max, brand_max) = con.execute(
        f"""
        SELECT count(*), max(user), max(cate), max(brand)
        FROM read_parquet('{behavior_cache_path.as_posix()}')
        """
    ).fetchone()
    max_sample_user = con.execute(
        f"SELECT max(user) FROM read_parquet('{static_sample_path.as_posix()}')"
    ).fetchone()[0]
    con.close()
    row_count = int(row_count)
    max_user = int(max(max_behavior_user, max_sample_user))
    print(f"Building shared behavior store: rows={row_count:,}, max_user={max_user:,}")
    cate_store = np.lib.format.open_memmap(
        tmp_dir / "cate.npy", mode="w+", dtype=np.int32, shape=(row_count,)
    )
    brand_store = np.lib.format.open_memmap(
        tmp_dir / "brand.npy", mode="w+", dtype=np.int32, shape=(row_count,)
    )
    btag_store = np.lib.format.open_memmap(
        tmp_dir / "btag.npy", mode="w+", dtype=np.int8, shape=(row_count,)
    )
    time_store = np.lib.format.open_memmap(
        tmp_dir / "time.npy", mode="w+", dtype=np.int64, shape=(row_count,)
    )
    user_start = np.full(max_user + 1, -1, dtype=np.int64)
    user_end = np.full(max_user + 1, -1, dtype=np.int64)
    parquet_file = pq.ParquetFile(behavior_cache_path)
    offset = 0
    previous_user = -1
    for batch_index, batch in enumerate(
        parquet_file.iter_batches(
            columns=["user", "time_stamp", "cate", "brand", "btag"],
            batch_size=STORE_BATCH_SIZE,
            use_threads=True,
        ),
        start=1,
    ):
        users = np.asarray(
            batch.column("user").to_numpy(zero_copy_only=False), dtype=np.int64
        )
        times = np.asarray(
            batch.column("time_stamp").to_numpy(zero_copy_only=False), dtype=np.int64
        )
        count = len(users)
        if count == 0:
            continue
        if previous_user > int(users[0]) or np.any(users[1:] < users[:-1]):
            raise RuntimeError("behavior cache is not globally sorted by user")
        previous_user = int(users[-1])
        cate_store[offset : offset + count] = np.asarray(
            batch.column("cate").to_numpy(zero_copy_only=False), dtype=np.int32
        )
        brand_store[offset : offset + count] = np.asarray(
            batch.column("brand").to_numpy(zero_copy_only=False), dtype=np.int32
        )
        btag_store[offset : offset + count] = np.asarray(
            batch.column("btag").to_numpy(zero_copy_only=False), dtype=np.int8
        )
        time_store[offset : offset + count] = times
        boundaries = np.flatnonzero(users[1:] != users[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [count]))
        for local_start, local_end in zip(starts, ends):
            user = int(users[local_start])
            if user_start[user] < 0:
                user_start[user] = offset + int(local_start)
            user_end[user] = offset + int(local_end)
        offset += count
        if batch_index % 25 == 0 or offset == row_count:
            print(
                f"Behavior store progress: {offset:,}/{row_count:,} ({offset / max(row_count, 1):.1%})"
            )
    if offset != row_count:
        raise RuntimeError(
            f"behavior store row mismatch: wrote={offset}, expected={row_count}"
        )
    for array in (cate_store, brand_store, btag_store, time_store):
        array.flush()
    del cate_store, brand_store, btag_store, time_store
    np.save(tmp_dir / "user_start.npy", user_start)
    np.save(tmp_dir / "user_end.npy", user_end)
    metadata = {
        "version": 1,
        "row_count": row_count,
        "max_user": max_user,
        "cate_id_max": int(cate_max or -1),
        "brand_max": int(brand_max or -1),
        "behavior_start_ts": BEHAVIOR_START_TS,
        "behavior_end_ts": BEHAVIOR_END_TS,
        "btag_encoding": {"unknown": 1, "pv": 2, "cart": 3, "fav": 4, "buy": 5},
        "strict_history": "behavior_time < exposure_time",
    }
    (tmp_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    remove_generated_path(store_dir)
    tmp_dir.rename(store_dir)
    print(f"Behavior store saved: {store_dir}")
    return metadata


def build_static_shards(
    static_sample_path: Path, shard_dir: Path, temp_dir: Path
) -> list[tuple[int, list[Path]]]:
    manifest_path = shard_dir / "manifest.json"
    if manifest_path.exists() and (not FORCE_REBUILD_POINTER):
        print(f"Reusing static shards: {shard_dir}")
    else:
        tmp_dir = shard_dir.with_name(f"{shard_dir.name}.tmp")
        remove_generated_path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=False)
        con = connect_duckdb(temp_dir)
        con.execute("PRAGMA preserve_insertion_order=false")
        con.execute(
            f"""
            COPY (
                SELECT *, CAST(hash(user) % {NUM_SHARDS} AS INTEGER) AS shard_id
                FROM read_parquet('{static_sample_path.as_posix()}')
            ) TO '{tmp_dir.as_posix()}' (
                FORMAT PARQUET,
                PARTITION_BY (shard_id),
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE {ROW_GROUP_SIZE}
            )
            """
        )
        row_count = con.execute(
            f"SELECT count(*) FROM read_parquet('{tmp_dir.as_posix()}/**/*.parquet', hive_partitioning=true)"
        ).fetchone()[0]
        con.close()
        (tmp_dir / "manifest.json").write_text(
            json.dumps(
                {"num_shards": NUM_SHARDS, "row_count": int(row_count)}, indent=2
            ),
            encoding="utf-8",
        )
        remove_generated_path(shard_dir)
        tmp_dir.rename(shard_dir)
        print(f"Static shards saved: {shard_dir}")
    shards: list[tuple[int, list[Path]]] = []
    for shard_id in range(NUM_SHARDS):
        paths = sorted((shard_dir / f"shard_id={shard_id}").glob("*.parquet"))
        if paths:
            shards.append((shard_id, paths))
    if not shards:
        raise RuntimeError(f"No static parquet shards found under {shard_dir}")
    return shards


def fill_keyed_features(
    exposure_times: np.ndarray,
    targets: np.ndarray,
    behavior_times: np.ndarray,
    behavior_keys: np.ndarray,
    behavior_btags: np.ndarray,
    prefix: str,
) -> dict[str, np.ndarray]:
    size = len(exposure_times)
    result = {
        f"same_{prefix}_cnt_1h": np.zeros(size, dtype=np.int32),
        f"same_{prefix}_cnt_6h": np.zeros(size, dtype=np.int32),
        f"same_{prefix}_cnt_1d": np.zeros(size, dtype=np.int32),
        f"hi_{prefix}_cnt_1d": np.zeros(size, dtype=np.int32),
        f"last_gap_{prefix}": np.full(size, -1, dtype=np.int64),
    }
    valid_targets = targets >= 0
    if not valid_targets.any() or len(behavior_times) == 0:
        return result
    order = np.argsort(behavior_keys, kind="stable")
    sorted_keys = behavior_keys[order]
    sorted_times = behavior_times[order]
    sorted_high_intent = np.isin(
        behavior_btags[order], np.array([3, 4, 5], dtype=np.int8)
    )
    for target in np.unique(targets[valid_targets]):
        exposure_indices = np.flatnonzero(targets == target)
        left = int(np.searchsorted(sorted_keys, target, side="left"))
        right = int(np.searchsorted(sorted_keys, target, side="right"))
        if right <= left:
            continue
        key_times = sorted_times[left:right]
        target_times = exposure_times[exposure_indices]
        history_end = np.searchsorted(key_times, target_times, side="left")
        for seconds, suffix in ((3600, "1h"), (21600, "6h"), (86400, "1d")):
            history_start = np.searchsorted(
                key_times, target_times - seconds, side="left"
            )
            result[f"same_{prefix}_cnt_{suffix}"][exposure_indices] = (
                history_end - history_start
            )
        high_times = key_times[sorted_high_intent[left:right]]
        high_end = np.searchsorted(high_times, target_times, side="left")
        high_start = np.searchsorted(high_times, target_times - 86400, side="left")
        result[f"hi_{prefix}_cnt_1d"][exposure_indices] = high_end - high_start
        has_recent = result[f"same_{prefix}_cnt_1d"][exposure_indices] > 0
        recent_indices = exposure_indices[has_recent]
        if len(recent_indices):
            last_positions = history_end[has_recent] - 1
            result[f"last_gap_{prefix}"][recent_indices] = (
                exposure_times[recent_indices] - key_times[last_positions]
            )
    return result


def process_static_shard(
    shard_id: int,
    input_paths: list[str],
    store_dir_str: str,
    output_path_str: str,
    max_behavior_len: int,
) -> dict:
    pa.set_cpu_count(1)
    tables = [pq.read_table(path, use_threads=False) for path in input_paths]
    table = pa.concat_tables(tables).combine_chunks()
    table = table.sort_by([("user", "ascending"), ("time_stamp", "ascending")])
    store_dir = Path(store_dir_str)
    behavior_time = np.load(store_dir / "time.npy", mmap_mode="r")
    behavior_cate = np.load(store_dir / "cate.npy", mmap_mode="r")
    behavior_brand = np.load(store_dir / "brand.npy", mmap_mode="r")
    behavior_btag = np.load(store_dir / "btag.npy", mmap_mode="r")
    user_start = np.load(store_dir / "user_start.npy", mmap_mode="r")
    user_end = np.load(store_dir / "user_end.npy", mmap_mode="r")
    users = np.asarray(
        table.column("user").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    exposure_times = np.asarray(
        table.column("time_stamp").to_numpy(zero_copy_only=False), dtype=np.int64
    )
    exposure_cate = np.asarray(
        table.column("cate_id").to_numpy(zero_copy_only=False), dtype=np.int32
    )
    exposure_brand = np.asarray(
        table.column("brand").to_numpy(zero_copy_only=False), dtype=np.int32
    )
    row_count = len(users)
    hist_start = np.zeros(row_count, dtype=np.int64)
    hist_end = np.zeros(row_count, dtype=np.int64)
    feature_arrays = {
        "same_cate_cnt_1h": np.zeros(row_count, dtype=np.int32),
        "same_cate_cnt_6h": np.zeros(row_count, dtype=np.int32),
        "same_cate_cnt_1d": np.zeros(row_count, dtype=np.int32),
        "hi_cate_cnt_1d": np.zeros(row_count, dtype=np.int32),
        "last_gap_cate": np.full(row_count, -1, dtype=np.int64),
        "same_brand_cnt_1h": np.zeros(row_count, dtype=np.int32),
        "same_brand_cnt_6h": np.zeros(row_count, dtype=np.int32),
        "same_brand_cnt_1d": np.zeros(row_count, dtype=np.int32),
        "hi_brand_cnt_1d": np.zeros(row_count, dtype=np.int32),
        "last_gap_brand": np.full(row_count, -1, dtype=np.int64),
    }
    boundaries = np.flatnonzero(users[1:] != users[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [row_count]))
    for exposure_start, exposure_end in zip(starts, ends):
        user = int(users[exposure_start])
        if user < 0 or user >= len(user_start) or user_start[user] < 0:
            continue
        behavior_start = int(user_start[user])
        behavior_end = int(user_end[user])
        user_times = np.asarray(behavior_time[behavior_start:behavior_end])
        local_exposure_times = exposure_times[exposure_start:exposure_end]
        local_hist_end = np.searchsorted(user_times, local_exposure_times, side="left")
        local_hist_start = np.maximum(0, local_hist_end - max_behavior_len)
        hist_start[exposure_start:exposure_end] = behavior_start + local_hist_start
        hist_end[exposure_start:exposure_end] = behavior_start + local_hist_end
        cate_features = fill_keyed_features(
            local_exposure_times,
            exposure_cate[exposure_start:exposure_end],
            user_times,
            np.asarray(behavior_cate[behavior_start:behavior_end]),
            np.asarray(behavior_btag[behavior_start:behavior_end]),
            "cate",
        )
        brand_features = fill_keyed_features(
            local_exposure_times,
            exposure_brand[exposure_start:exposure_end],
            user_times,
            np.asarray(behavior_brand[behavior_start:behavior_end]),
            np.asarray(behavior_btag[behavior_start:behavior_end]),
            "brand",
        )
        for name, values in (*cate_features.items(), *brand_features.items()):
            feature_arrays[name][exposure_start:exposure_end] = values
    seq_len = (hist_end - hist_start).astype(np.int32)
    output = table.append_column("hist_start", pa.array(hist_start, type=pa.int64()))
    output = output.append_column("hist_end", pa.array(hist_end, type=pa.int64()))
    output = output.append_column("seq_len", pa.array(seq_len, type=pa.int32()))
    for prefix in ("cate", "brand"):
        for suffix in ("1h", "6h", "1d"):
            count_name = f"same_{prefix}_cnt_{suffix}"
            hit_name = f"same_{prefix}_hit_{suffix}"
            output = output.append_column(
                hit_name,
                pa.array(
                    (feature_arrays[count_name] > 0).astype(np.int8), type=pa.int8()
                ),
            )
            output = output.append_column(
                count_name, pa.array(feature_arrays[count_name], type=pa.int32())
            )
        high_count_name = f"hi_{prefix}_cnt_1d"
        output = output.append_column(
            f"hi_{prefix}_hit_1d",
            pa.array(
                (feature_arrays[high_count_name] > 0).astype(np.int8), type=pa.int8()
            ),
        )
        output = output.append_column(
            high_count_name, pa.array(feature_arrays[high_count_name], type=pa.int32())
        )
        output = output.append_column(
            f"last_gap_{prefix}",
            pa.array(feature_arrays[f"last_gap_{prefix}"], type=pa.int64()),
        )
    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        output,
        output_path,
        compression=PARQUET_COMPRESSION,
        use_dictionary=True,
        write_statistics=True,
        row_group_size=ROW_GROUP_SIZE,
    )
    seq_hist = np.bincount(seq_len, minlength=max_behavior_len + 1)
    return {
        "shard_id": shard_id,
        "rows": row_count,
        "users": len(starts),
        "seq_hist": seq_hist.tolist(),
    }


def histogram_quantile(histogram: np.ndarray, quantile: float) -> int:
    threshold = max(1, int(np.ceil(histogram.sum() * quantile)))
    return int(np.searchsorted(np.cumsum(histogram), threshold, side="left"))


def build_pointer_dataset(
    shards: list[tuple[int, list[Path]]], store_dir: Path, output_dir: Path
) -> dict:
    manifest_path = output_dir / "_manifest.json"
    if manifest_path.exists() and (not FORCE_REBUILD_POINTER):
        print(f"Reusing pointer dataset: {output_dir}")
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    tmp_dir = output_dir.with_name(f"{output_dir.name}.tmp")
    remove_generated_path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    results = []
    with ProcessPoolExecutor(max_workers=POINTER_WORKERS) as executor:
        futures = {
            executor.submit(
                process_static_shard,
                shard_id,
                [path.as_posix() for path in paths],
                store_dir.as_posix(),
                (tmp_dir / f"part-{shard_id:05d}.parquet").as_posix(),
                2048,
            ): shard_id
            for (shard_id, paths) in shards
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"Pointer shard progress: {completed}/{len(futures)}, shard={result['shard_id']}, rows={result['rows']:,}"
            )
    results.sort(key=lambda item: item["shard_id"])
    seq_hist = np.sum(
        [np.asarray(item["seq_hist"], dtype=np.int64) for item in results], axis=0
    )
    row_count = int(sum((item["rows"] for item in results)))
    manifest = {
        "version": 1,
        "row_count": row_count,
        "num_shards": len(results),
        "max_seq_len": 2048,
        "behavior_store_path": store_dir.as_posix(),
        "strict_history": "behavior_time < exposure_time",
        "sequence": {
            "mean": float(
                np.dot(np.arange(len(seq_hist)), seq_hist) / max(seq_hist.sum(), 1)
            ),
            "nonzero_ratio": float(1.0 - seq_hist[0] / max(seq_hist.sum(), 1)),
            "p50": histogram_quantile(seq_hist, 0.5),
            "p75": histogram_quantile(seq_hist, 0.75),
            "p90": histogram_quantile(seq_hist, 0.9),
            "max": int(np.flatnonzero(seq_hist > 0)[-1]) if seq_hist.any() else 0,
        },
    }
    (tmp_dir / "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    remove_generated_path(output_dir)
    tmp_dir.rename(output_dir)
    print(f"Pointer dataset saved: {output_dir}")
    print(json.dumps(manifest, indent=2))
    return manifest


def validate_config() -> None:
    if NUM_SHARDS <= 0 or POINTER_WORKERS <= 0:
        raise RuntimeError(
            "DIN_POINTER_SHARDS and DIN_POINTER_WORKERS must be positive"
        )


def build() -> None:
    validate_config()
    temp_dir = ensure_directories()
    (static_sample_path, behavior_cache_path) = build_paths()
    (store_dir, static_shard_dir, output_dir) = pointer_paths()
    print("DIN pointer build configuration:")
    print(f"max_behavior_len={2048}")
    print(f"behavior_time_range=[{BEHAVIOR_START_TS}, {BEHAVIOR_END_TS}]")
    print(f"num_shards={NUM_SHARDS}, pointer_workers={POINTER_WORKERS}")
    print(
        f"force_static={FORCE_REBUILD_STATIC}, force_behavior={FORCE_REBUILD_BEHAVIOR}"
    )
    print(f"force_pointer={FORCE_REBUILD_POINTER}")
    build_static_sample(static_sample_path, temp_dir)
    build_behavior_cache(behavior_cache_path, static_sample_path, temp_dir)
    build_behavior_store(behavior_cache_path, static_sample_path, store_dir, temp_dir)
    shards = build_static_shards(static_sample_path, static_shard_dir, temp_dir)
    build_pointer_dataset(shards, store_dir, output_dir)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--memory-limit", default="8GB")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    global \
        RAW_DIR, \
        PROCESSED_DIR, \
        SAMPLE_DIR, \
        RAW_SAMPLE_PATH, \
        AD_PATH, \
        USER_PATH, \
        BEHAVIOR_PATH
    global \
        NUM_SHARDS, \
        POINTER_WORKERS, \
        DUCKDB_THREADS, \
        DUCKDB_MEMORY_LIMIT, \
        FORCE_REBUILD_STATIC, \
        FORCE_REBUILD_BEHAVIOR, \
        FORCE_REBUILD_POINTER
    data = args.data_dir.resolve()
    (RAW_DIR, PROCESSED_DIR, SAMPLE_DIR) = (
        data / "raw_data",
        data / "processed_data",
        data / "sample",
    )
    (RAW_SAMPLE_PATH, AD_PATH, USER_PATH) = [
        RAW_DIR / name
        for name in ["raw_sample.csv", "ad_feature.csv", "user_profile.csv"]
    ]
    BEHAVIOR_PATH = PROCESSED_DIR / "behavior_log.parquet"
    NUM_SHARDS = POINTER_WORKERS = args.workers
    DUCKDB_THREADS = args.threads
    DUCKDB_MEMORY_LIMIT = args.memory_limit
    FORCE_REBUILD_STATIC = FORCE_REBUILD_BEHAVIOR = FORCE_REBUILD_POINTER = args.rebuild
    if not BEHAVIOR_PATH.exists():
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        con = connect_duckdb(ensure_directories())
        source = RAW_DIR / "behavior_log.csv"
        con.execute(
            f"COPY (SELECT * FROM read_csv_auto('{source.as_posix()}', header=true)) TO '{BEHAVIOR_PATH.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        con.close()
    build()


if __name__ == "__main__":
    main()
