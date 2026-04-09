"""
Test NVFP4 KV cache calibration (TENSOR_GROUP strategy, group_size=16).

NVFP4 KV cache uses two-level scaling:
  - Global scale (FP32): calibrated per-tensor over the dataset (static)
  - Per-group scale (FP8, group_size=16): computed dynamically at runtime

With dynamic=DynamicType.LOCAL, calibration only produces k_global_scale
and v_global_scale. The per-group k_scale/v_scale are NOT created because
they are computed on-the-fly during inference.
"""

import torch
import torch.nn as nn
from compressed_tensors.quantization import (
    DynamicType,
    QuantizationArgs,
    QuantizationStatus,
    QuantizationStrategy,
    apply_quantization_config,
    is_attention_module,
)
from compressed_tensors.quantization.quant_args import FP8_E4M3_DATA
from transformers import PretrainedConfig

from llmcompressor.modifiers.quantization.calibration import (
    calibrate_key_hook,
    calibrate_value_hook,
)
from llmcompressor.modifiers.quantization.quantization import QuantizationModifier

HEAD_DIM = 64
NUM_HEADS = 2
GROUP_SIZE = 16
HIDDEN_DIM = HEAD_DIM * NUM_HEADS


class _StubAttention(nn.Module):
    """Minimal attention module recognized by is_attention_module()."""

    def __init__(self, dim: int = HIDDEN_DIM, head_dim: int = HEAD_DIM):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.o_proj(self.q_proj(x) + self.k_proj(x) + self.v_proj(x))


class _StubBlock(nn.Module):
    def __init__(self, dim: int = HIDDEN_DIM):
        super().__init__()
        self.self_attn = _StubAttention(dim)
        self.mlp = nn.Linear(dim, dim)

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class _StubModel(nn.Module):
    def __init__(
        self,
        dim: int = HIDDEN_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = 1,
    ):
        super().__init__()
        self.config = PretrainedConfig(
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            hidden_size=dim,
        )
        self.config._attn_implementation = "eager"
        self.layers = nn.ModuleList([_StubBlock(dim) for _ in range(num_layers)])

    def set_attn_implementation(self, implementation: str):
        self.config._attn_implementation = implementation

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _make_nvfp4_kv_scheme():
    return QuantizationArgs(
        num_bits=4,
        type="float",
        strategy=QuantizationStrategy.TENSOR_GROUP,
        symmetric=True,
        dynamic=DynamicType.LOCAL,
        group_size=GROUP_SIZE,
        observer="static_minmax",
        scale_dtype=FP8_E4M3_DATA.dtype,
        zp_dtype=FP8_E4M3_DATA.dtype,
    )


def test_nvfp4_kv_cache_initialization():
    """Verify that TENSOR_GROUP + DynamicType.LOCAL creates only
    k_global_scale / v_global_scale (no per-group k_scale / v_scale)."""
    model = _StubModel()
    modifier = QuantizationModifier(
        targets=["Linear"],
        kv_cache_scheme=_make_nvfp4_kv_scheme(),
    )

    apply_quantization_config(model, modifier.resolved_config)

    attn_modules = [
        (name, m) for name, m in model.named_modules() if is_attention_module(m)
    ]
    assert len(attn_modules) > 0

    for name, m in attn_modules:
        assert hasattr(
            m, "quantization_scheme"
        ), f"quantization_scheme not set on {name}"
        # Global scales must exist (calibrated per-tensor)
        assert hasattr(m, "k_global_scale"), f"k_global_scale missing on {name}"
        assert hasattr(m, "v_global_scale"), f"v_global_scale missing on {name}"
        assert m.k_global_scale.shape == (
            1,
        ), f"k_global_scale should be scalar, got {m.k_global_scale.shape}"
        assert m.v_global_scale.shape == (
            1,
        ), f"v_global_scale should be scalar, got {m.v_global_scale.shape}"
        # Per-group scales must NOT exist (dynamic=LOCAL skips them)
        assert not hasattr(
            m, "k_scale"
        ), f"k_scale should not exist on {name} with dynamic=LOCAL"
        assert not hasattr(
            m, "v_scale"
        ), f"v_scale should not exist on {name} with dynamic=LOCAL"


def test_nvfp4_kv_cache_calibration_lifecycle():
    """Full lifecycle: init → start_calibration → feed data → end_calibration.
    Only k_global_scale / v_global_scale should be calibrated (per-tensor)."""
    model = _StubModel()
    modifier = QuantizationModifier(
        targets=["Linear"],
        kv_cache_scheme=_make_nvfp4_kv_scheme(),
    )

    apply_quantization_config(model, modifier.resolved_config)
    modifier.start_calibration(model)

    attn_modules = [
        (name, m) for name, m in model.named_modules() if is_attention_module(m)
    ]

    # Verify observers were created
    for name, m in attn_modules:
        assert m.quantization_status == QuantizationStatus.CALIBRATION
        assert hasattr(m, "k_observer"), f"k_observer missing on {name}"
        assert hasattr(m, "v_observer"), f"v_observer missing on {name}"

    # Simulate calibration by feeding fake key/value tensors through the hooks
    batch_size, seq_len = 2, 32
    for _, m in attn_modules:
        fake_k = torch.randn(batch_size, NUM_HEADS, seq_len, HEAD_DIM)
        fake_v = torch.randn(batch_size, NUM_HEADS, seq_len, HEAD_DIM)
        calibrate_key_hook(m, fake_k)
        calibrate_value_hook(m, fake_v)

    modifier.end_calibration(model)

    for name, m in attn_modules:
        assert m.quantization_status == QuantizationStatus.FROZEN
        assert not hasattr(m, "k_observer"), "k_observer should be removed"
        assert not hasattr(m, "v_observer"), "v_observer should be removed"

        # global_scale must be a positive finite FP32 scalar
        k_gs = m.k_global_scale
        v_gs = m.v_global_scale
        assert k_gs.dtype == torch.float32, f"k_global_scale dtype {k_gs.dtype}"
        assert v_gs.dtype == torch.float32, f"v_global_scale dtype {v_gs.dtype}"
        assert k_gs.shape == (1,), f"k_global_scale shape {k_gs.shape}"
        assert v_gs.shape == (1,), f"v_global_scale shape {v_gs.shape}"
        assert torch.isfinite(k_gs).all(), f"k_global_scale not finite on {name}"
        assert torch.isfinite(v_gs).all(), f"v_global_scale not finite on {name}"
        assert (k_gs > 0).all(), f"k_global_scale not positive on {name}"
        assert (v_gs > 0).all(), f"v_global_scale not positive on {name}"

        # Per-group scales must NOT exist (dynamic at runtime)
        assert not hasattr(
            m, "k_scale"
        ), "k_scale should not exist after calibration with dynamic=LOCAL"
        assert not hasattr(
            m, "v_scale"
        ), "v_scale should not exist after calibration with dynamic=LOCAL"


def test_nvfp4_kv_flatten_attention_shape():
    """Verify that _flatten_attention produces correct shape for TENSOR_GROUP.
    This reshape is used by the static TENSOR_GROUP path; in the NVFP4 KV case
    (dynamic=LOCAL) only the global scale path runs, but the reshape must still
    be correct for completeness."""
    from llmcompressor.observers.helpers import flatten_for_calibration

    args = QuantizationArgs(
        num_bits=4,
        type="float",
        strategy=QuantizationStrategy.TENSOR_GROUP,
        symmetric=True,
        dynamic=False,
        group_size=GROUP_SIZE,
    )
    batch_size, num_heads, seq_len, head_dim = 4, NUM_HEADS, 64, HEAD_DIM
    value = torch.randn(batch_size, num_heads, seq_len, head_dim)

    flat = flatten_for_calibration(value, "k", args)

    expected_shape = (
        batch_size * seq_len,
        num_heads,
        head_dim // GROUP_SIZE,
        GROUP_SIZE,
    )
    assert flat.shape == expected_shape, f"Got {flat.shape}, expected {expected_shape}"


def test_q_fp8_with_nvfp4_kv_calibration():
    """Verify that q_scheme (FP8 per-tensor) can be calibrated independently
    of kv_cache_scheme (NVFP4 global scale)."""
    from llmcompressor.modifiers.quantization.calibration import (
        calibrate_query_hook,
    )

    fp8_q_args = QuantizationArgs(
        num_bits=8,
        type="float",
        strategy=QuantizationStrategy.TENSOR,
        symmetric=True,
        dynamic=False,
    )

    model = _StubModel()
    modifier = QuantizationModifier(
        targets=["Linear"],
        kv_cache_scheme=_make_nvfp4_kv_scheme(),
        q_scheme=fp8_q_args,
    )

    apply_quantization_config(model, modifier.resolved_config)
    modifier.initialize_quantization(model)
    modifier.start_calibration(model)

    attn_modules = [
        (name, m) for name, m in model.named_modules() if is_attention_module(m)
    ]

    for name, m in attn_modules:
        # q should have its own observer using FP8 args
        assert hasattr(m, "q_observer"), f"q_observer missing on {name}"
        # k/v should have observers using NVFP4 args
        assert hasattr(m, "k_observer"), f"k_observer missing on {name}"
        assert hasattr(m, "v_observer"), f"v_observer missing on {name}"

        # q_scale should exist (FP8 per-tensor, static)
        assert hasattr(m, "q_scale"), f"q_scale missing on {name}"
        assert m.q_scale.shape == (1,), "q_scale should be per-tensor scalar"

        # k/v should only have global_scale (NVFP4, dynamic=LOCAL)
        assert hasattr(m, "k_global_scale"), f"k_global_scale missing on {name}"
        assert not hasattr(m, "k_scale"), "k_scale should not exist (dynamic=LOCAL)"

    # Simulate calibration
    batch_size, seq_len = 2, 32
    for _, m in attn_modules:
        fake_q = torch.randn(batch_size, NUM_HEADS, seq_len, HEAD_DIM)
        fake_k = torch.randn(batch_size, NUM_HEADS, seq_len, HEAD_DIM)
        fake_v = torch.randn(batch_size, NUM_HEADS, seq_len, HEAD_DIM)
        calibrate_query_hook(m, fake_q)
        calibrate_key_hook(m, fake_k)
        calibrate_value_hook(m, fake_v)

    modifier.end_calibration(model)

    for name, m in attn_modules:
        assert m.quantization_status == QuantizationStatus.FROZEN
        assert not hasattr(m, "q_observer")
        assert not hasattr(m, "k_observer")

        # q: per-tensor FP8 scale — finite positive scalar
        assert torch.isfinite(m.q_scale).all()
        assert m.q_scale.shape == (1,)

        # k/v: NVFP4 global scale — finite positive scalar
        assert torch.isfinite(m.k_global_scale).all()
        assert (m.k_global_scale > 0).all()
        assert torch.isfinite(m.v_global_scale).all()
        assert (m.v_global_scale > 0).all()


def test_group_strategy_rejected_for_attention():
    """Plain GROUP strategy must still be rejected for attention tensors."""
    import pytest

    from llmcompressor.observers.helpers import flatten_for_calibration

    args = QuantizationArgs(
        num_bits=4,
        type="float",
        strategy=QuantizationStrategy.GROUP,
        symmetric=True,
        dynamic=False,
        group_size=GROUP_SIZE,
    )
    value = torch.randn(2, NUM_HEADS, 32, HEAD_DIM)
    with pytest.raises(ValueError, match="Group quantization cannot be applied"):
        flatten_for_calibration(value, "k", args)
