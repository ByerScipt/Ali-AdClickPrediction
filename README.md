# Ad Click-Through Rate Prediction

CTR prediction on the Alibaba display advertising dataset (Tianchi 56), with **26.6M ad impressions** and **723.1M behavior events**. This project adapts MixFormer to combine semantic feature groups with user behavior sequences and compares it with LightGBM, DCNv2, DCNv2 + DIN, and RankMixer + DIN.

## Models

- **LightGBM** uses user, ad, and context features with historical click and behavior aggregates.
- **DCNv2** uses two low-rank cross layers over tabular features. **DCNv2 + DIN** adds target-conditioned attention over user history.
- **RankMixer + DIN** groups tabular features into semantic tokens and adds a DIN interest token.
- **MixFormer** organizes tabular inputs into 16 semantic heads across two mixing blocks. Each block mixes feature heads and uses cross-attention to read behavior events preceding the impression.

Sequence models use the latest 1,024 events, represented by category, brand, behavior type, and timestamps. All neural models exclude user and ad-group IDs from their embeddings. Their tabular features include behavior counts over 1 hour, 6 hours, and 1 day, plus recency features. LightGBM uses the 1-hour and 1-day aggregates.

## Results

Impressions are split chronologically in **UTC+8 (Asia/Shanghai)**: May 6–11, 2017 for training (20,015,245 rows), May 12 for validation (3,234,051), and May 13 for testing (3,308,665). Checkpoints are selected by validation AUC. All runs use seed 42; neural models were trained with four GPUs and BF16 DDP.

| Model | Validation AUC | Test AUC | Test LogLoss |
| :--- | ---: | ---: | ---: |
| LightGBM | 0.671454 | 0.669192 | **0.190683** |
| DCNv2 | 0.671687 | 0.669470 | 0.191392 |
| DCNv2 + DIN | 0.675709 | 0.673920 | 0.191470 |
| RankMixer + DIN | 0.676255 | 0.674150 | 0.192637 |
| MixFormer | **0.678074** | **0.676430** | 0.191204 |

MixFormer improves test AUC by **0.0072** over LightGBM and **0.0025** over DCNv2 + DIN. LightGBM has the lowest LogLoss. Against RankMixer + DIN, its test AUC gain is 0.002280 (paired DeLong 95% CI: [0.001906, 0.002655]). This compares the selected checkpoints on the same test impressions; it does not measure variation across training seeds.

### Ablations

These runs use the MixFormer configuration and are compared on validation AUC.

| Variant | Validation AUC |
| :--- | ---: |
| MixFormer | **0.678074** |
| Without sequence cross-attention | 0.672998 |
| Without head mixing | 0.611638 |

The first ablation zeros the cross-attention residual; the second replaces head mixing with an identity operation.

## How to start

Use Python 3.10+ and install the dependencies:

```bash
pip install -r requirements.txt
```

### Prepare the data

Download the [Alibaba display advertising dataset](https://tianchi.aliyun.com/dataset/56). Extract `raw_sample.csv`, `ad_feature.csv`, `user_profile.csv`, and `behavior_log.csv` into `data/raw_data/`, then run:

```bash
python -m DataEng.prepare --data-dir data --workers 8 --threads 8 --memory-limit 8GB
```

Preparation joins user and ad attributes, computes historical exposure statistics, and builds a shared behavior store with per-impression history pointers. Behavior timestamps must be strictly earlier than the impression. Exposure aggregates use labels from earlier rows in timestamp order, including earlier impressions on validation and test days. Dense features are standardized using training dates only. Prepared histories retain up to 2,048 events; model inputs use the most recent 1,024.

The builder reuses existing caches. Pass `--rebuild` to regenerate prepared features and pointers. An existing `data/processed_data/behavior_log.parquet` can be used in place of `behavior_log.csv`.

### Train and evaluate

```bash
python -m baseline.LGBM --data-dir data

torchrun --standalone --nproc_per_node=4 --module behavior.train \
  --config configs/mixformer.json --data-dir data

python -m behavior.train --config configs/mixformer.json \
  --data-dir data --evaluate test
```

Use `configs/dcn.json`, `configs/dcn_din.json`, or `configs/rankmixer_din.json` for the neural baselines. Each configuration includes the training settings used for that model. Batch sizes are per GPU: 1,024 for MixFormer and 4,096 for the neural baselines. MixFormer validates every 1,000 steps; the baselines validate after each epoch.

Training writes `best.pt`, `last.pt`, and `history.json` to `outputs/<config name>/`. Evaluation writes predictions and metrics to the same directory. Use `--output-dir` for a separate run, `--resume` to resume from `last.pt` at the next epoch, or `--checkpoint` to evaluate a specific checkpoint. For CPU execution, add `--cpu` and use smaller batch and sequence lengths in the configuration.

Run the ablations with separate output directories:

```bash
torchrun --standalone --nproc_per_node=4 --module behavior.train \
  --config configs/mixformer.json --no-sequence-cross-attention \
  --output-dir outputs/no_sequence_cross_attention

torchrun --standalone --nproc_per_node=4 --module behavior.train \
  --config configs/mixformer.json --no-head-mixing \
  --output-dir outputs/no_head_mixing
```

To compare two evaluation exports generated with the same data and evaluation layout:

```bash
python -m behavior.compare outputs/rankmixer_din/test.npz outputs/mixformer/test.npz \
  --output outputs/comparison.json
```

## References

- [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)
- [DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems](https://arxiv.org/abs/2008.13535)
- [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)
- [RankMixer](https://arxiv.org/abs/2507.15551)
- [MixFormer](https://arxiv.org/abs/2602.14110)
