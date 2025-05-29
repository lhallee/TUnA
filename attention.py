import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Optional, Tuple
from functools import partial
from einops import rearrange
from rotary import RotaryEmbedding, RotaryEmbeddingCross


Linear = partial(nn.Linear, bias=False)


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout, rotary=True, use_spectral_norm=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_head = hidden_size // n_heads
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"

        if use_spectral_norm:
            self.w_q = spectral_norm(Linear(hidden_size, hidden_size))
            self.w_k = spectral_norm(Linear(hidden_size, hidden_size))
            self.w_v = spectral_norm(Linear(hidden_size, hidden_size))
            self.out_proj = spectral_norm(Linear(hidden_size, hidden_size))
        else:
            self.layernorm_qkv = nn.Sequential(
                nn.LayerNorm(hidden_size), Linear(hidden_size, hidden_size * 3)
            )
            self.out_proj = Linear(hidden_size, hidden_size)
        self.use_spectral_norm = use_spectral_norm
        if not use_spectral_norm:
            self.q_ln = nn.LayerNorm(hidden_size, bias=False)
            self.k_ln = nn.LayerNorm(hidden_size, bias=False)
        self.dropout_rate = dropout
        self.rotary = RotaryEmbedding(hidden_size // n_heads) if rotary else None
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, L, _ = x.shape
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :].expand(b, 1, L, L).bool()
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1).bool()
            elif attention_mask.dim() == 4:
                attention_mask = attention_mask.bool()
            else:
                raise ValueError(f"Invalid attention mask dimension: {attention_mask.dim()}")

        if self.use_spectral_norm:
            q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        else:
            qkv = self.layernorm_qkv(x)
            q, k, v = qkv.chunk(3, dim=-1)

        if not self.use_spectral_norm:
            q, k = self.q_ln(q).to(q.dtype), self.k_ln(k).to(q.dtype)

        if self.rotary:
            q, k = self._apply_rotary(q, k)

        q, k, v = map(self.reshaper, (q, k, v))
        a = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.dropout_rate,
            is_causal=False
        )
        a = rearrange(a, "b h s d -> b s (h d)") # (bs, seq_len, n_heads * d_head)
        return self.out_proj(a) # (bs, seq_len, hidden_size)


class AttentionPooler(nn.Module):
    """
    Cross-attention mechanism for pooling (b, L, d) -> (b, n_tokens, d_pooled)
    """
    def __init__(
            self,
            hidden_size: int,
            n_tokens: int = 1,
            n_heads: int = 16,
            use_spectral_norm: bool = True
    ):
        super(AttentionPooler, self).__init__()
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"
        self.n_tokens = n_tokens
        self.d_head = hidden_size // n_heads
        self.Q = nn.Parameter(torch.randn(1, n_tokens, hidden_size))
        self.Wq = Linear(hidden_size, hidden_size) if use_spectral_norm else Linear(hidden_size, hidden_size)
        self.Wv = Linear(hidden_size, hidden_size) if use_spectral_norm else Linear(hidden_size, hidden_size)
        self.Wk = Linear(hidden_size, hidden_size) if use_spectral_norm else Linear(hidden_size, hidden_size)
        corrected_hidden = (hidden_size // n_heads) * n_heads
        self.Wo = Linear(corrected_hidden, hidden_size) if use_spectral_norm else Linear(corrected_hidden, hidden_size)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, L, d = x.size()
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :].expand(b, 1, self.n_tokens, L).bool()
        q = self.Wq(self.Q).expand(b, -1, -1)  # (b, n_tokens, d)
        v = self.Wv(x)  # (b, L, d)
        k = self.Wk(x)  # (b, L, d)
        q, k, v = map(self.reshaper, (q, k, v))  # (b, n_heads, n_tokens, d_head) (b, n_heads, L, d_head)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=False
        ) # (b, n_heads, n_tokens, d_head)
        attn = rearrange(attn, "b h s d -> b s (h d)")  # (b, n_tokens, n_heads * d_head)
        return self.Wo(attn)  # (b, n_tokens, d_pooled)
    

class CrossAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, rotary: bool = True, causal: bool = False, use_spectral_norm: bool = True):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.d_head = self.hidden_size // self.n_heads
        self.Wq = spectral_norm(Linear(hidden_size, hidden_size)) if use_spectral_norm else Linear(hidden_size, hidden_size)
        self.Wk = spectral_norm(Linear(hidden_size, hidden_size)) if use_spectral_norm else Linear(hidden_size, hidden_size)
        self.Wv = spectral_norm(Linear(hidden_size, hidden_size)) if use_spectral_norm else Linear(hidden_size, hidden_size)
        corrected_hidden = (hidden_size // n_heads) * n_heads
        self.out_proj = spectral_norm(Linear(corrected_hidden, hidden_size)) if use_spectral_norm else Linear(corrected_hidden, hidden_size)
        self.use_spectral_norm = use_spectral_norm
        if not use_spectral_norm:
            self.q_ln = nn.LayerNorm(hidden_size, bias=False)
            self.k_ln = nn.LayerNorm(hidden_size, bias=False)
        self.reshaper = partial(rearrange, pattern="b s (h d) -> b h s d", h=n_heads)
        self.rotary = RotaryEmbeddingCross(hidden_size // n_heads) if rotary else None
        self.causal = causal

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            attention_mask_1: Optional[torch.Tensor] = None,
            attention_mask_2: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        # For cross attention, we need a mask that goes from L1 (query) to L2 (key)
        attention_mask = None
        if attention_mask_1 is not None and attention_mask_2 is not None:
            # Create cross attention mask: queries from x1 attending to keys from x2
            # Shape: (bs, 1, L1, L2)
            attention_mask = torch.einsum('bi,bj->bij', attention_mask_1, attention_mask_2)
            attention_mask = attention_mask.unsqueeze(1).bool()
        q = self.Wq(x1) # (bs, L1, hidden_size)
        k = self.Wk(x2) # (bs, L2, hidden_size)
        v = self.Wv(x2) # (bs, L2, hidden_size)
        if not self.use_spectral_norm:
            q, k = self.q_ln(q).to(q.dtype), self.k_ln(k).to(q.dtype)
        if self.rotary:
            q, k = self._apply_rotary(q, k)
        q, k, v = map(self.reshaper, (q, k, v)) # (bs, n_heads, L1, d_head) (bs, n_heads, L2, d_head) (bs, n_heads, L2, d_head)
        a = F.scaled_dot_product_attention(q, k, v, attention_mask if not self.causal else None, is_causal=self.causal) # (bs, n_heads, L1, d_head)
        a = rearrange(a, "b h s d -> b s (h d)") # (bs, L1, n_heads * d_head)
        return self.out_proj(a) # (bs, L1, hidden_size)
