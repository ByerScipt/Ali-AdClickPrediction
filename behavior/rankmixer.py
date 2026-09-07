from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class MultiHeadTokenMixing(nn.Module):
    def __init__(self, model_dim: int, token_count: int) -> None:
        super().__init__()
        if token_count <= 0 or model_dim % token_count:
            raise ValueError("model_dim must be divisible by the positive token_count")
        self.model_dim = model_dim
        self.token_count = token_count
        self.head_dim = model_dim // token_count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected (batch, tokens, dim), got {tuple(x.shape)}")
        (batch, tokens, dim) = x.shape
        if tokens != self.token_count or dim != self.model_dim:
            raise ValueError(
                f"expected (*, {self.token_count}, {self.model_dim}), got {tuple(x.shape)}"
            )
        return (
            x.reshape(batch, tokens, self.token_count, self.head_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch, tokens, dim)
        )


class PerTokenFFN(nn.Module):
    def __init__(
        self, model_dim: int, token_count: int, expansion_ratio: int = 4
    ) -> None:
        super().__init__()
        if expansion_ratio <= 0:
            raise ValueError("expansion_ratio must be positive")
        hidden_dim = model_dim * expansion_ratio
        self.model_dim = model_dim
        self.token_count = token_count
        self.weight1 = nn.Parameter(torch.empty(token_count, model_dim, hidden_dim))
        self.bias1 = nn.Parameter(torch.zeros(token_count, hidden_dim))
        self.weight2 = nn.Parameter(torch.empty(token_count, hidden_dim, model_dim))
        self.bias2 = nn.Parameter(torch.zeros(token_count, model_dim))
        nn.init.xavier_uniform_(self.weight1)
        nn.init.xavier_uniform_(self.weight2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.size(1) != self.token_count or x.size(2) != self.model_dim:
            raise ValueError(
                f"expected (*, {self.token_count}, {self.model_dim}), got {tuple(x.shape)}"
            )
        hidden = torch.einsum("btd,tdh->bth", x, self.weight1) + self.bias1
        hidden = torch.nn.functional.gelu(hidden)
        return torch.einsum("bth,thd->btd", hidden, self.weight2) + self.bias2


class RankMixerBlock(nn.Module):
    def __init__(
        self,
        model_dim: int,
        token_count: int,
        expansion_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_mixing = MultiHeadTokenMixing(model_dim, token_count)
        self.token_norm = nn.LayerNorm(model_dim)
        self.pffn = PerTokenFFN(model_dim, token_count, expansion_ratio)
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = self.dropout(self.token_mixing(x))
        x = self.token_norm(mixed + x)
        ffn = self.dropout(self.pffn(x))
        return self.ffn_norm(ffn + x)


class RankMixerBackbone(nn.Module):
    def __init__(
        self,
        model_dim: int,
        token_count: int,
        num_layers: int = 4,
        expansion_ratio: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.model_dim = model_dim
        self.token_count = token_count
        self.blocks = nn.ModuleList(
            [
                RankMixerBlock(model_dim, token_count, expansion_ratio, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            tokens = block(tokens)
        return tokens

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(tokens)
        return tokens.mean(dim=1)


@dataclass(frozen=True)
class TokenGroup:
    name: str
    fields: tuple[str, ...]


SEMANTIC_V1_SPARSE_GROUPS = (
    TokenGroup("placement", ("pid",)),
    TokenGroup("candidate_category", ("cate_id",)),
    TokenGroup("candidate_brand", ("brand",)),
    TokenGroup("audience_group", ("cms_group_id",)),
    TokenGroup("audience_segment", ("cms_segid",)),
    TokenGroup("user_gender", ("final_gender_code",)),
    TokenGroup("user_occupation", ("occupation",)),
    TokenGroup("request_time", ("hour", "weekday")),
)
SEMANTIC_V1_DENSE_GROUPS = (
    TokenGroup("candidate_price", ("price",)),
    TokenGroup("user_ordinal", ("age_level", "shopping_level")),
    TokenGroup("ad_history", ("ad_hist_imp", "ad_hist_clk")),
    TokenGroup(
        "user_global_history", ("user_hist_imp", "user_hist_clk", "user_hist_ctr")
    ),
    TokenGroup(
        "user_category_history",
        ("user_cate_hist_imp", "user_cate_hist_clk", "user_cate_hist_ctr"),
    ),
    TokenGroup(
        "target_match_short_horizon",
        (
            "same_cate_hit_1h",
            "same_cate_cnt_1h",
            "same_brand_hit_1h",
            "same_brand_cnt_1h",
            "same_cate_hit_6h",
            "same_cate_cnt_6h",
            "same_brand_hit_6h",
            "same_brand_cnt_6h",
        ),
    ),
    TokenGroup(
        "target_match_day_horizon",
        (
            "same_cate_hit_1d",
            "same_cate_cnt_1d",
            "same_brand_hit_1d",
            "same_brand_cnt_1d",
        ),
    ),
    TokenGroup(
        "high_interest_recency",
        (
            "hi_cate_hit_1d",
            "hi_cate_cnt_1d",
            "hi_brand_hit_1d",
            "hi_brand_cnt_1d",
            "last_gap_cate",
            "last_gap_brand",
        ),
    ),
)


def _validate_exact_coverage(
    kind: str, available_fields: list[str], groups: tuple[TokenGroup, ...]
) -> None:
    grouped_fields = [field for group in groups for field in group.fields]
    duplicate_fields = sorted(
        {field for field in grouped_fields if grouped_fields.count(field) > 1}
    )
    missing_fields = sorted(set(available_fields) - set(grouped_fields))
    unexpected_fields = sorted(set(grouped_fields) - set(available_fields))
    if duplicate_fields or missing_fields or unexpected_fields:
        raise ValueError(
            f"semantic_v1 {kind} token layout does not exactly cover the active feature schema; duplicates={duplicate_fields}, missing={missing_fields}, unexpected={unexpected_fields}"
        )


def semantic_v1_groups(
    sparse_cols: list[str], dense_cols: list[str]
) -> tuple[tuple[TokenGroup, ...], tuple[TokenGroup, ...]]:
    _validate_exact_coverage("sparse", sparse_cols, SEMANTIC_V1_SPARSE_GROUPS)
    _validate_exact_coverage("dense", dense_cols, SEMANTIC_V1_DENSE_GROUPS)
    return (SEMANTIC_V1_SPARSE_GROUPS, SEMANTIC_V1_DENSE_GROUPS)


def describe_rankmixer_layout(
    tokenization: str, sparse_cols: list[str], dense_cols: list[str]
) -> list[dict[str, object]]:
    if tokenization == "field":
        groups = [
            {"kind": "sparse", "name": field, "fields": [field]}
            for field in sparse_cols
        ]
        groups.append(
            {
                "kind": "dense",
                "name": "all_dense_statistics",
                "fields": list(dense_cols),
            }
        )
        return groups
    if tokenization == "semantic_v1":
        (sparse_groups, dense_groups) = semantic_v1_groups(sparse_cols, dense_cols)
        return [
            *[
                {"kind": "sparse", "name": group.name, "fields": list(group.fields)}
                for group in sparse_groups
            ],
            *[
                {"kind": "dense", "name": group.name, "fields": list(group.fields)}
                for group in dense_groups
            ],
        ]
    raise ValueError(f"unsupported RankMixer tokenization: {tokenization}")
