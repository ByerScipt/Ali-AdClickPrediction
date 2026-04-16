# Corrected-Date Experiment Report (2026-04-15)

## 1. Root Cause Audit

### 1.1 Official date range

- Raw `time_stamp` interpreted in `UTC` gives `2017-05-05 ~ 2017-05-13`
- Raw `time_stamp` interpreted in `Asia/Shanghai (UTC+8)` gives `2017-05-06 ~ 2017-05-13`
- The public Ali-CCP description is consistent with the `UTC+8` interpretation

### 1.2 What was wrong

- Old `WideDeepSample.parquet` and `LGBMSample.parquet` were bucketed by `UTC`
- Therefore they incorrectly contained `2017-05-05`
- Old `DINSample_full.parquet` already matched the official `UTC+8` day buckets

### 1.3 Code fixes

- `DataEng/WideDeepFeat.py`
  - changed time extraction from raw UTC truncation to `time_stamp + 8h`
  - changed `date` to real `Date` instead of day-truncated `Datetime`
- `DataEng/LGBMFeat.py`
  - changed pandas time conversion to `utc=True -> Asia/Shanghai`
- `DataEng/DeepFeat.py`
  - changed pandas time conversion to `utc=True -> Asia/Shanghai`
- `DataEng/DINFeat.py`
  - restored default `DIN_TIME_SHIFT_HOURS=0`
  - `shiftm8h` is now only for historical reproduction, not the default path

### 1.4 Corrected day counts

The corrected `WideDeepSample.parquet`, corrected `LGBMSample.parquet`, and official `DINSample_full.parquet` now all match:

- `2017-05-06: 3,270,348`
- `2017-05-07: 3,430,002`
- `2017-05-08: 3,354,523`
- `2017-05-09: 3,248,016`
- `2017-05-10: 3,333,752`
- `2017-05-11: 3,378,604`
- `2017-05-12: 3,234,051`
- `2017-05-13: 3,308,665`

## 2. Model-Code Alignment Fixes

### 2.1 DIN / DIN+DCN feature semantics

- `age_level` and `shopping_level` were corrected to `dense`
- `DIN.py` no longer treats them as sparse embeddings
- `DIN_DCN.py` default path priority was restored to official `DINSample_full.parquet`

### 2.2 Runtime optimization

- `DIN.py`
  - added `bf16` AMP support
  - added gradient clipping
  - added `DIN_DISABLE_TQDM`
  - increased scanner readahead
  - added epoch timing into history
- `DIN_DCN.py`
  - already had streaming parquet training, `bf16`, batch-order shuffle, epoch timing
- `baseline/LGBM.py`
  - added configurable hyperparameters and early stopping
  - added `n_jobs`
- `deep/DCN.py` / `deep/WideDeep.py`
  - added configurable data path and date-range printing

## 3. Official-Corrected Results

All results below use:

- train: `2017-05-06 ~ 2017-05-12`
- valid: `2017-05-13`

### 3.1 LGBM

Sanity rerun with old weak settings:

- log: `lgbm.log`
- `n_estimators=45`
- `num_leaves=3`
- `AUC=0.651743`
- `LogLoss=0.192569`

Fair tuned reruns:

- `leaf31`
  - log: `lgbm_tune_leaf31.log`
  - best iteration: `129`
  - `AUC=0.664283`
  - `LogLoss=0.191003`
- `leaf63`
  - log: `lgbm_tune_leaf63.log`
  - best iteration: `110`
  - `AUC=0.664519`
  - `LogLoss=0.190979`
- `leaf127`
  - log: `lgbm_tune_leaf127.log`
  - best iteration: `95`
  - `AUC=0.664286`
  - `LogLoss=0.191006`

Current corrected best-tuned LGBM:

- `AUC=0.664519`
- `LogLoss=0.190979`
- config: `num_leaves=63`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`

### 3.2 DCN

- log: `dcn_low_rank32.log`
- config: low-rank `rank=32`
- best epoch: `6`
- `AUC=0.665778`
- `LogLoss=0.192122`

### 3.3 Wide&Deep

These are now secondary, but were rerun after the date fix:

- `no_cross`
  - log: `widedeep_nocross.log`
  - best epoch: `4`
  - `AUC=0.630911`
  - `LogLoss=0.201458`
- `full_cross`
  - log: `widedeep_fullcross.log`
  - best epoch: `8`
  - `AUC=0.617876`
  - `LogLoss=0.215988`

### 3.4 Pure DIN

- log: `din_official_bs16384.log`
- config:
  - official `DINSample_full.parquet`
  - `include_user=0`
  - `age_level/shopping_level` as dense
  - `batch=16384`
- manual stop after epoch `4` because gains were flattening and time cost was already high
- best observed:
  - `AUC=0.6441`
  - `LogLoss=0.1933`

### 3.5 DIN+DCN (seq100)

- log: `dindcn_official_seq100.log`
- config:
  - `parallel`
  - low-rank `rank=32`
  - `cross_feature_mode=small`
  - `include_adgroup=1`
  - `age_level/shopping_level` as dense
  - `seq_len=100`
- manual stop after epoch `4` to move to longer-sequence verification
- best observed:
  - `AUC=0.6541`
  - `LogLoss=0.1923`

### 3.6 DIN+DCN (seq256)

- log: `dindcn_official_seq256.log`
- data: `DINSample_full_seq256.parquet`
- sequence statistics:
  - mean `227.81`
  - `p25/p50/p75/p90 = 256`
- best:
  - epoch `3`
  - `AUC=0.661541`
  - `LogLoss=0.191905`

## 4. Time-Cost Comparison

Approximate epoch cost on this server:

- corrected `DCN(rank=32)`: first epoch `72.70s`, then about `6.6 ~ 9.1s/epoch`
- pure `DIN`:
  - pilot `batch=4096`: `374.87s/epoch`
  - optimized `batch=16384`: `260 ~ 270s/epoch`
- `DIN+DCN seq100`: about `316 ~ 319s/epoch`
- `DIN+DCN seq256`: about `531 ~ 579s/epoch`

Implication:

- `DIN` and `DIN+DCN` are much more expensive than `DCN`
- longer sequence does help, but the extra time cost is very large

## 5. Interpretation

### 5.1 What was disproved

- It was **not** valid to compare old `DCN/WideDeep/LGBM` against official `DIN` before fixing the date buckets
- The earlier `2017-05-05` split was a real data bug

### 5.2 What was verified

- `age_level / shopping_level` being sparse was a genuine semantic bug and is now fixed
- `seq_len=100` was too short for this dataset
- increasing to `256` raises `DIN+DCN` from `0.6541` to `0.6615`

### 5.3 Why DIN+DCN still did not beat corrected DCN

Most likely combined reasons:

- current task is `clk` only, while Ali-CCP public usage is often discussed together with click+conversion settings
- the current `DCN` already has strong static statistics and explicit crosses, so the incremental value of DIN is smaller
- even after improvement, `seq256` still has much higher time cost than `DCN`
- the current DIN branch is still a simplified engineering version, not a full reproduction of every DIN training trick

### 5.4 Practical conclusion

Under the corrected official date split:

- best current `DCN` is still the best pure deep baseline
- best current `LGBM` is also very strong and competitive
- `DIN+DCN` is not useless:
  - it improves materially when the sequence is lengthened
  - but it currently does **not** beat corrected `DCN`
  - and its time cost is much worse

## 6. Files / Logs

- corrected-date logs directory:
  - `/root/autodl-tmp/AliCCP/behavior/checkpoints/corrected_date_rerun_20260415`
- key result logs:
  - `dcn_low_rank32.log`
  - `lgbm_tune_leaf31.log`
  - `lgbm_tune_leaf63.log`
  - `lgbm_tune_leaf127.log`
  - `din_official_bs16384.log`
  - `dindcn_official_seq100.log`
  - `dindcn_official_seq256.log`
