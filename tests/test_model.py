import torch
import pytest
from llm_scratch.model.layers import RollingKV, RMSNorm, RoPECache, apply_rope_single

def test_rolling_kv_keep_window_with_sink():
    B,H,D = 1,2,4
    kv = RollingKV(window=4, sink=2)
    for _ in range(10):
        k_new = torch.randn(B,H,1,D)
        v_new = torch.randn(B,H,1,D)
        k,v = kv.step(k_new, v_new)
        # Should never exceed sink+window length
        assert k.size(2) <= 6

def test_rmsnorm_shapes():
    x = torch.randn(2,3,8)
    rn = RMSNorm(8)
    y = rn(x)
    assert y.shape == x.shape

def test_rope_rotation_shapes_single():
    # Vanilla case: same #heads for q and k
    B, H, T, D = 1, 2, 5, 8
    rc = RoPECache(head_dim=D, max_pos=32)
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    pos = torch.arange(0, T)
    cos, sin = rc.get(pos)

    q2 = apply_rope_single(q, cos, sin)
    k2 = apply_rope_single(k, cos, sin)

    assert q2.shape == q.shape
    assert k2.shape == k.shape
    # check that rotation mixes even/odd features (values should change)
    assert not torch.allclose(q2, q)
    assert not torch.allclose(k2, k)

def test_rope_rotation_shapes_gqa():
    # GQA case: q has H heads; k has fewer Hk heads (shared KV)
    B, H, Hk, T, D = 2, 8, 2, 7, 16
    rc = RoPECache(head_dim=D, max_pos=128)
    q = torch.randn(B, H,  T, D)
    k = torch.randn(B, Hk, T, D)
    pos = torch.arange(10, 10 + T)  # arbitrary start position
    cos, sin = rc.get(pos)

    q2 = apply_rope_single(q, cos, sin)
    k2 = apply_rope_single(k, cos, sin)

    assert q2.shape == (B, H,  T, D)
    assert k2.shape == (B, Hk, T, D)
    # rotations should be deterministic for same positions
    # check a couple of coords moved as expected (values changed)
    assert not torch.allclose(q2, q)
    assert not torch.allclose(k2, k)
