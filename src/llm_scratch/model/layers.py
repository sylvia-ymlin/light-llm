from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# -----------------------------------------------------------------------------
# KV Cache
# -----------------------------------------------------------------------------

@dataclass
class KVCache:
    k: torch.Tensor  # (B,H,T,D)
    v: torch.Tensor  # (B,H,T,D)

    @property
    def T(self):
        return self.k.size(2)

class RollingKV:
    """Rolling buffer with optional attention sink.
    Keeps first `sink` tokens + last `window` tokens.
    """
    def __init__(self, window: int, sink: int = 0):
        self.window = window
        self.sink = sink
        self.k = None
        self.v = None
    
    def step(self, k_new: torch.Tensor, v_new: torch.Tensor):
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        # crop
        if self.k.size(2) > self.window + self.sink:
            sink_part = self.k[:, :, :self.sink, :]
            sink_val  = self.v[:, :, :self.sink, :]
            tail_k = self.k[:, :, -self.window:, :]
            tail_v = self.v[:, :, -self.window:, :]
            self.k = torch.cat([sink_part, tail_k], dim=2)
            self.v = torch.cat([sink_val, tail_v], dim=2)
        return self.k, self.v

# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    y = x * g / rms(x),   rms(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight

# -----------------------------------------------------------------------------
# Feed Forward
# -----------------------------------------------------------------------------

class SwiGLU(nn.Module):
    """SwiGLU FFN: (xW1) ⊗ swish(xW2) W3  with expansion factor `mult`.
    """
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = mult * dim
        self.w1 = nn.Linear(dim, inner, bias=False)
        self.w2 = nn.Linear(dim, inner, bias=False)
        self.w3 = nn.Linear(inner, dim, bias=False)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        a = self.w1(x)
        b = self.act(self.w2(x))
        return self.drop(self.w3(a * b))

# -----------------------------------------------------------------------------
# Positional Embeddings (RoPE)
# -----------------------------------------------------------------------------

class RoPECache:
    """Precompute cos/sin for positions up to max_pos for even head_dim."""
    def __init__(self, head_dim: int, max_pos: int, base: float = 10000.0, device: torch.device | None = None):
        assert head_dim % 2 == 0, "RoPE head_dim must be even"
        self.head_dim = head_dim
        self.base = base
        self.device = device
        self._build(max_pos)
        
    def get(self, positions: torch.Tensor):
        # positions: (T,) or (1,T)
        if positions.dim() == 2:
            positions = positions[0]
        need = int(positions.max().item()) + 1 if positions.numel() > 0 else 1
        if need > self.max_pos:
            # grow tables
            self._build(max(need, int(self.max_pos * 2)))
        cos = self.cos[positions]  # (T, D/2)
        sin = self.sin[positions]
        return cos, sin
    
    def _build(self, max_pos: int):
        """(Re)build cos/sin tables for a new max_pos."""
        self.max_pos = max_pos
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, device=self.device).float() / self.head_dim))
        t = torch.arange(max_pos, device=self.device).float()
        freqs = torch.outer(t, inv_freq)  # (max_pos, head_dim/2)
        self.cos = torch.cos(freqs)
        self.sin = torch.sin(freqs)

def apply_rope_single(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate pairs along last dim for RoPE.
    x: (B,H,T,D) with D even; cos/sin: (T,D/2)
    """
    assert x.size(-1) % 2 == 0
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1,1,T,D/2)
    sin = sin.unsqueeze(0).unsqueeze(0)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    xr1 = x1 * cos - x2 * sin
    xr2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., ::2] = xr1
    out[..., 1::2] = xr2
    return out

# -----------------------------------------------------------------------------
# Attention
# -----------------------------------------------------------------------------

class CausalSelfAttentionModern(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0,
                 rope: bool = True, max_pos: int = 4096,
                 sliding_window: int | None = None, attention_sink: int = 0,
                 n_kv_head: int | None = None):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = n_head
        self.n_kv_head = n_kv_head or n_head
        assert self.n_head % self.n_kv_head == 0, "n_head must be multiple of n_kv_head (GQA grouping)"
        self.group_size = self.n_head // self.n_kv_head
        self.d_head = n_embd // n_head

        # Separate projections for Q vs K/V (sizes differ under GQA)
        self.wq  = nn.Linear(n_embd, self.n_head   * self.d_head, bias=False)
        self.wk  = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        self.wv  = nn.Linear(n_embd, self.n_kv_head * self.d_head, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.use_rope = rope
        self.rope_cache: RoPECache | None = None
        self.max_pos = max_pos
        self.sliding_window = sliding_window
        self.attention_sink = attention_sink

    def _maybe_init_rope(self, device):
        if self.use_rope and self.rope_cache is None:
            self.rope_cache = RoPECache(self.d_head, self.max_pos, device=device)

    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None, start_pos: int = 0):
        """x: (B,T,C). If kv_cache given, we assume generation (T small, often 1)."""
        B, T, C = x.shape
        self._maybe_init_rope(x.device)

        # Projections
        q = self.wq(x).view(B, T, self.n_head,   self.d_head).transpose(1, 2)    # (B,H, T,D)
        k = self.wk(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)   # (B,Hk,T,D)
        v = self.wv(x).view(B, T, self.n_kv_head, self.d_head).transpose(1, 2)   # (B,Hk,T,D)

        # RoPE on *current* tokens (cached keys are already rotated)
        if self.use_rope:
            pos = torch.arange(start_pos, start_pos + T, device=x.device)
            cos, sin = self.rope_cache.get(pos)
            q = apply_rope_single(q, cos, sin)   # (B,H, T,D)
            k = apply_rope_single(k, cos, sin)   # (B,Hk,T,D)

        # Concatenate past cache (cache is stored in Hk heads)
        if kv_cache is not None:
            k_all = torch.cat([kv_cache.k, k], dim=2)  # (B,Hk, Tpast+T, D)
            v_all = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_all, v_all = k, v

        # Sliding-window + attention-sink (crop along seq length)
        if self.sliding_window is not None and k_all.size(2) > (self.sliding_window + self.attention_sink):
            s = self.attention_sink
            k_all = torch.cat([k_all[:, :, :s, :], k_all[:, :, -self.sliding_window:, :]], dim=2)
            v_all = torch.cat([v_all[:, :, :s, :], v_all[:, :, -self.sliding_window:, :]], dim=2)

        # --- GQA expand: repeat K/V heads to match Q heads before attention ---
        if self.n_kv_head != self.n_head:
            k_attn = k_all.repeat_interleave(self.group_size, dim=1)  # (B,H,Tk,D)
            v_attn = v_all.repeat_interleave(self.group_size, dim=1)  # (B,H,Tk,D)
        else:
            k_attn, v_attn = k_all, v_all

        # Scaled dot-product attention (PyTorch scales internally)
        is_causal = kv_cache is None
        y = F.scaled_dot_product_attention(q, k_attn, v_attn,
                                           attn_mask=None,
                                           dropout_p=self.dropout.p if self.training else 0.0,
                                           is_causal=is_causal)          # (B,H,T,D)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        # Update KV cache (store compact Hk heads, not expanded)
        if kv_cache is not None:
            k_new = torch.cat([kv_cache.k, k], dim=2)  # (B,Hk,*,D)
            v_new = torch.cat([kv_cache.v, v], dim=2)
        else:
            k_new, v_new = k, v
        new_cache = KVCache(k_new, v_new)
        return y, new_cache
