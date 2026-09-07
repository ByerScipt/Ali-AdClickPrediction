from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-06) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("RMSNorm dim must be positive")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        inverse_rms = torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (x_float * inverse_rms * self.weight).to(input_dtype)


class SwiGLUFFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("SwiGLU input_dim and hidden_dim must be positive")
        self.gate = nn.Linear(input_dim, hidden_dim, bias=False)
        self.up = nn.Linear(input_dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class PerHeadSwiGLUFFN(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, hidden_dim: int) -> None:
        super().__init__()
        if num_heads <= 0 or head_dim <= 0 or hidden_dim <= 0:
            raise ValueError("PerHeadSwiGLUFFN dimensions must be positive")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim
        self.gate = nn.Parameter(torch.empty(num_heads, head_dim, hidden_dim))
        self.up = nn.Parameter(torch.empty(num_heads, head_dim, hidden_dim))
        self.down = nn.Parameter(torch.empty(num_heads, hidden_dim, head_dim))
        for parameter in (self.gate, self.up, self.down):
            for head_index in range(num_heads):
                nn.init.xavier_uniform_(parameter[head_index])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = (self.num_heads, self.head_dim)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"expected (batch, {expected[0]}, {expected[1]}), got {tuple(x.shape)}"
            )
        gate = torch.einsum("bnd,ndh->bnh", x, self.gate)
        up = torch.einsum("bnd,ndh->bnh", x, self.up)
        hidden = F.silu(gate) * up
        return torch.einsum("bnh,nhd->bnd", hidden, self.down)


def head_mixing(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError(f"expected (batch, heads, dim), got {tuple(x.shape)}")
    (batch_size, num_heads, head_dim) = x.shape
    if num_heads <= 0 or head_dim % num_heads:
        raise ValueError(
            f"MixFormer HeadMixing requires head_dim divisible by heads, got {head_dim}/{num_heads}"
        )
    sub_dim = head_dim // num_heads
    return (
        x.reshape(batch_size, num_heads, num_heads, sub_dim)
        .transpose(1, 2)
        .reshape(batch_size, num_heads, head_dim)
    )


class QueryMixer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        ffn_hidden_dim: int,
        norm_eps: float,
        use_head_mixing: bool = True,
    ) -> None:
        super().__init__()
        self.use_head_mixing = use_head_mixing
        self.norm_before_mix = RMSNorm(head_dim, norm_eps)
        self.norm_before_ffn = RMSNorm(head_dim, norm_eps)
        self.ffn = PerHeadSwiGLUFFN(num_heads, head_dim, ffn_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_before_mix(x)
        mixed_input = head_mixing(normalized) if self.use_head_mixing else normalized
        mixed = mixed_input + x
        return self.ffn(self.norm_before_ffn(mixed)) + mixed


class CrossAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        sequence_ffn_hidden_dim: int,
        norm_eps: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = num_heads * head_dim
        self.scale = head_dim ** (-0.5)
        self.sequence_norm = RMSNorm(self.total_dim, norm_eps)
        self.sequence_ffn = SwiGLUFFN(self.total_dim, sequence_ffn_hidden_dim)
        self.key = nn.Parameter(torch.empty(num_heads, head_dim, head_dim))
        self.value = nn.Parameter(torch.empty(num_heads, head_dim, head_dim))
        for parameter in (self.key, self.value):
            for head_index in range(num_heads):
                nn.init.xavier_uniform_(parameter[head_index])
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self, query: torch.Tensor, sequence: torch.Tensor, sequence_mask: torch.Tensor
    ) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.size(-1) != self.total_dim:
            raise ValueError(
                f"expected sequence (batch, length, {self.total_dim}), got {tuple(sequence.shape)}"
            )
        if sequence_mask.ndim != 2 or sequence_mask.shape != sequence.shape[:2]:
            raise ValueError(
                f"mask {tuple(sequence_mask.shape)} does not match sequence {tuple(sequence.shape[:2])}"
            )
        (batch_size, sequence_length, _) = sequence.shape
        transformed = self.sequence_ffn(self.sequence_norm(sequence)) + sequence
        transformed = transformed.reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        )
        key = torch.einsum("btnd,nde->btne", transformed, self.key)
        value = torch.einsum("btnd,nde->btne", transformed, self.value)
        score = torch.einsum("bnd,btnd->bnt", query, key).float() * self.scale
        mask = sequence_mask.unsqueeze(1)
        score = score.masked_fill(~mask, torch.finfo(score.dtype).min)
        weight = F.softmax(score, dim=-1)
        weight = weight * mask.to(weight.dtype)
        weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(weight.dtype).eps
        )
        weight = self.attn_dropout(weight).to(value.dtype)
        aggregated = torch.einsum("bnt,btnd->bnd", weight, value)
        return aggregated + query


class OutputFusion(nn.Module):
    def __init__(
        self, num_heads: int, head_dim: int, ffn_hidden_dim: int, norm_eps: float
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(head_dim, norm_eps)
        self.ffn = PerHeadSwiGLUFFN(num_heads, head_dim, ffn_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.norm(x)) + x


class MixFormerBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        head_ffn_hidden_dim: int,
        sequence_ffn_hidden_dim: int,
        norm_eps: float,
        dropout: float,
        use_head_mixing: bool = True,
        use_sequence_cross_attention: bool = True,
    ) -> None:
        super().__init__()
        self.use_sequence_cross_attention = use_sequence_cross_attention
        self.query_mixer = QueryMixer(
            num_heads,
            head_dim,
            head_ffn_hidden_dim,
            norm_eps,
            use_head_mixing=use_head_mixing,
        )
        self.cross_attention = CrossAttention(
            num_heads, head_dim, sequence_ffn_hidden_dim, norm_eps, dropout
        )
        self.output_fusion = OutputFusion(
            num_heads, head_dim, head_ffn_hidden_dim, norm_eps
        )

    def forward(
        self, x: torch.Tensor, sequence: torch.Tensor, sequence_mask: torch.Tensor
    ) -> torch.Tensor:
        query = self.query_mixer(x)
        attended = self.cross_attention(query, sequence, sequence_mask)
        if self.use_sequence_cross_attention:
            query = attended
        else:
            query = query + (attended - query) * 0.0
        return self.output_fusion(query)


class MixFormerCTRBackbone(nn.Module):
    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        adapter_dim: int,
        sequence_input_dim: int,
        num_layers: int,
        head_ffn_expansion: int,
        sequence_ffn_expansion: int,
        task_hidden_dims: Sequence[int],
        dropout: float,
        norm_eps: float = 1e-06,
        gradient_checkpointing: bool = False,
        use_head_mixing: bool = True,
        use_sequence_cross_attention: bool = True,
    ) -> None:
        super().__init__()
        if (
            num_heads <= 0
            or head_dim <= 0
            or adapter_dim <= 0
            or (sequence_input_dim <= 0)
        ):
            raise ValueError("MixFormer dimensions must be positive")
        if num_layers <= 0 or head_ffn_expansion <= 0 or sequence_ffn_expansion <= 0:
            raise ValueError("MixFormer depth and FFN expansions must be positive")
        if head_dim % num_heads:
            raise ValueError(
                f"MixFormer requires head_dim divisible by num_heads for HeadMixing, got {head_dim}/{num_heads}"
            )
        if any((dim <= 0 for dim in task_hidden_dims)):
            raise ValueError("task_hidden_dims must contain positive dimensions")
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.adapter_dim = adapter_dim
        self.total_dim = num_heads * head_dim
        self.gradient_checkpointing = gradient_checkpointing
        self.use_head_mixing = use_head_mixing
        self.use_sequence_cross_attention = use_sequence_cross_attention
        self.nonsequential_projection = nn.Parameter(
            torch.empty(num_heads, adapter_dim, head_dim)
        )
        for head_index in range(num_heads):
            nn.init.xavier_uniform_(self.nonsequential_projection[head_index])
        self.sequence_projection = nn.Linear(sequence_input_dim, self.total_dim)
        head_hidden_dim = head_dim * head_ffn_expansion
        sequence_hidden_dim = self.total_dim * sequence_ffn_expansion
        self.blocks = nn.ModuleList(
            [
                MixFormerBlock(
                    num_heads,
                    head_dim,
                    head_hidden_dim,
                    sequence_hidden_dim,
                    norm_eps,
                    dropout,
                    use_head_mixing=use_head_mixing,
                    use_sequence_cross_attention=use_sequence_cross_attention,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(head_dim, norm_eps)
        prediction_layers: list[nn.Module] = []
        current_dim = self.total_dim
        for hidden_dim in task_hidden_dims:
            prediction_layers.append(nn.Linear(current_dim, hidden_dim))
            prediction_layers.append(nn.ReLU())
            if dropout > 0:
                prediction_layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        prediction_layers.append(nn.Linear(current_dim, 1))
        self.task_head = nn.Sequential(*prediction_layers)

    def forward(
        self,
        nonsequential_tokens: torch.Tensor,
        sequence_events: torch.Tensor,
        sequence_lengths: torch.Tensor,
    ) -> torch.Tensor:
        expected_tokens = (self.num_heads, self.adapter_dim)
        if (
            nonsequential_tokens.ndim != 3
            or tuple(nonsequential_tokens.shape[1:]) != expected_tokens
        ):
            raise ValueError(
                f"expected nonsequential tokens (batch, {expected_tokens[0]}, {expected_tokens[1]}), got {tuple(nonsequential_tokens.shape)}"
            )
        if (
            sequence_events.ndim != 3
            or sequence_events.size(-1) != self.sequence_projection.in_features
        ):
            raise ValueError(
                f"expected sequence events (batch, length, {self.sequence_projection.in_features}), got {tuple(sequence_events.shape)}"
            )
        if sequence_events.size(0) != nonsequential_tokens.size(0):
            raise ValueError("non-sequential and sequential batch sizes differ")
        x = torch.einsum(
            "bna,nad->bnd", nonsequential_tokens, self.nonsequential_projection
        )
        sequence = self.sequence_projection(sequence_events)
        positions = torch.arange(sequence.size(1), device=sequence.device).unsqueeze(0)
        sequence_mask = positions < sequence_lengths.unsqueeze(1)
        sequence = sequence * sequence_mask.unsqueeze(-1).to(sequence.dtype)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    lambda x_value, sequence_value: block(
                        x_value, sequence_value, sequence_mask
                    ),
                    x,
                    sequence,
                    use_reentrant=False,
                )
            else:
                x = block(x, sequence, sequence_mask)
        x = self.final_norm(x).reshape(x.size(0), -1)
        return self.task_head(x).squeeze(-1)
