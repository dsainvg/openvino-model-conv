"""Convert Qwen3.5-4B to OpenVINO IR in split-IR format (layer-by-layer).

Strategy mirrors qwen36/scripts/convert_toy.py:
  - Feed each PyTorch wrapper module DIRECTLY to ov.convert_model()
    (OpenVINO's Torch FX capture) instead of going through torch.onnx.export.
  - This avoids ALL ONNX exporter version/Dynamo issues.
  - Layers are converted one at a time to keep RAM low.

Output layout:
  <output_dir>/embed.xml          -- embedding lookup
  <output_dir>/layer_N.xml        -- one per decoder layer (full or linear attn)
  <output_dir>/lm_head.xml        -- final norm + lm_head projection
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# ── Kill OV telemetry before it can crash in the threaded Kaggle environment ──
try:
    import openvino_telemetry
    def _noop(*a, **kw):
        pass
    openvino_telemetry.Telemetry.send_event    = _noop
    openvino_telemetry.Telemetry.start_session = _noop
    openvino_telemetry.Telemetry.end_session   = _noop
    openvino_telemetry.Telemetry.send_error    = _noop
    openvino_telemetry.Telemetry.send_stack_trace = _noop
    try:
        import openvino_telemetry.utils.sender
        openvino_telemetry.utils.sender.TelemetrySender.send = _noop
    except Exception:
        pass
except ImportError:
    pass

import openvino as ov

from src.configuration import Qwen35Config
from src.load_weights import (
    build_shard_index,
    load_layer_weights,
    _find_key,
    _load_safetensors_weight,
    load_linear_weight,
)
from src.modeling import QwenDecoderLayer, QwenForCausalLM
from src.split_inference import (
    QwenEmbedWrapper,
    QwenLayerFullWrapper,
    QwenLayerLinearWrapper,
    QwenLMHeadWrapper,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def copy_non_weights_files(src_dir: Path, dst_dir: Path) -> None:
    import shutil
    ignore_suffixes = {".safetensors", ".bin", ".pt", ".ckpt", ".h5", ".msgpack", ".ot"}
    for item in src_dir.iterdir():
        if item.is_file() and item.suffix.lower() not in ignore_suffixes:
            shutil.copy2(item, dst_dir / item.name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Qwen3.5-4B to split-IR OpenVINO.")
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Path to the Hugging Face weights directory.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output directory for split IR files.")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16",
                   help="Dtype to process weights (default bf16).")
    p.add_argument("--compile-check", action="store_true",
                   help="Sequentially compile each IR on CPU to verify correctness.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    output_dir = args.output.resolve() if args.output else (REPO.parent / "ov_ir_qwen35_4b")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir  = args.model_dir.resolve()

    out_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(f"Model dir   : {model_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Weight dtype: {args.dtype}")

    cfg        = Qwen35Config.from_pretrained_dir(model_dir)
    weight_map = build_shard_index(model_dir)

    # ──────────────────────────────────────────────────────────────
    # 1. Embedding
    # ──────────────────────────────────────────────────────────────
    print("\n[1/3] Converting Embedding...")
    shell = QwenForCausalLM(cfg)

    embed_key = _find_key(weight_map, "embed_tokens.weight", [
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "model.embed_tokens.weight",
    ])
    shell.model.embed_tokens.weight.data.copy_(
        _load_safetensors_weight(weight_map, model_dir, embed_key, out_dtype)
    )

    embed_wrapper = QwenEmbedWrapper(shell).eval()
    # seq_len > 1 for shape inference: OV captures a richer dynamic shape.
    example_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    ov_embed = ov.convert_model(embed_wrapper, example_input=(example_ids,))
    ov.save_model(ov_embed, str(output_dir / "embed.xml"), compress_to_fp16=True)
    del shell, embed_wrapper, ov_embed
    gc.collect()
    print("  embed.xml ✓")

    # ──────────────────────────────────────────────────────────────
    # 2. Decoder Layers  (one at a time → low RAM)
    # ──────────────────────────────────────────────────────────────
    print(f"\n[2/3] Converting {cfg.num_hidden_layers} decoder layers sequentially...")
    for i in range(cfg.num_hidden_layers):
        t0 = time.time()
        lt = cfg.layer_types[i]
        print(f"  [{i+1}/{cfg.num_hidden_layers}] layer_{i}  ({lt}) ...", flush=True)

        layer = QwenDecoderLayer(cfg, layer_idx=i).eval()
        load_layer_weights(layer, i, weight_map, model_dir, out_dtype=out_dtype)

        if lt == "full_attention":
            wrapper = QwenLayerFullWrapper(layer).eval()
            x         = torch.randn(1, 1, cfg.hidden_size,                                      dtype=out_dtype)
            cos        = torch.randn(1, 1, cfg.partial_rotary_dim,                               dtype=out_dtype)
            sin        = torch.randn(1, 1, cfg.partial_rotary_dim,                               dtype=out_dtype)
            k_cache    = torch.zeros(1, cfg.num_key_value_heads, 8, cfg.head_dim,               dtype=out_dtype)
            v_cache    = torch.zeros_like(k_cache)
            write_pos  = torch.tensor(0, dtype=torch.long)
            example_in = (x, cos, sin, k_cache, v_cache, write_pos)

        else:  # linear_attention
            wrapper = QwenLayerLinearWrapper(layer).eval()
            x         = torch.randn(1, 1, cfg.hidden_size,                                                   dtype=out_dtype)
            conv       = torch.zeros(1, cfg.linear_conv_dim, cfg.linear_conv_kernel_dim,                     dtype=out_dtype)
            rec        = torch.zeros(1, cfg.linear_num_value_heads,
                                     cfg.linear_key_head_dim, cfg.linear_value_head_dim,                     dtype=out_dtype)
            example_in = (x, conv, rec)

        ov_layer = ov.convert_model(wrapper, example_input=example_in)
        ov.save_model(ov_layer, str(output_dir / f"layer_{i}.xml"), compress_to_fp16=True)

        del layer, wrapper, ov_layer
        gc.collect()
        print(f"    layer_{i}.xml ✓  ({time.time()-t0:.1f}s)")

    # ──────────────────────────────────────────────────────────────
    # 3. LM Head + final norm
    # ──────────────────────────────────────────────────────────────
    print("\n[3/3] Converting LM Head + final norm...")
    shell = QwenForCausalLM(cfg)

    norm_key = _find_key(weight_map, "norm.weight", [
        "model.language_model.norm.weight",
        "language_model.model.norm.weight",
        "model.norm.weight",
    ])
    try:
        lm_key = _find_key(weight_map, "lm_head.weight", [
            "lm_head.weight",
            "model.language_model.lm_head.weight",
            "language_model.lm_head.weight",
            "lm_head.qweight",
            "model.language_model.lm_head.qweight",
            "language_model.lm_head.qweight",
        ])
    except KeyError:
        lm_key = _find_key(weight_map, "lm_head.qweight", [
            "lm_head.qweight",
            "model.language_model.lm_head.qweight",
            "language_model.lm_head.qweight",
        ])

    shell.model.norm.weight.data.copy_(
        _load_safetensors_weight(weight_map, model_dir, norm_key, out_dtype)
    )

    lm_prefix = lm_key.removesuffix(".qweight").removesuffix(".weight")
    shell.lm_head.weight.data.copy_(
        load_linear_weight(weight_map, model_dir, lm_prefix, out_dtype)
    )

    lm_wrapper = QwenLMHeadWrapper(shell).eval()
    example_x  = torch.randn(1, 3, cfg.hidden_size, dtype=out_dtype)

    ov_lm = ov.convert_model(lm_wrapper, example_input=(example_x,))
    ov.save_model(ov_lm, str(output_dir / "lm_head.xml"), compress_to_fp16=True)
    del shell, lm_wrapper, ov_lm
    gc.collect()
    print("  lm_head.xml ✓")

    # Copy tokenizer / config files alongside the IR
    copy_non_weights_files(model_dir, output_dir)
    print(f"\n✓ Split-IR conversion complete → {output_dir}")

    # ──────────────────────────────────────────────────────────────
    # Optional: sequential compile-check (one model at a time)
    # ──────────────────────────────────────────────────────────────
    if args.compile_check:
        print("\nCompile-check: loading each IR on CPU sequentially...")
        core = ov.Core()
        print("  embed ...")
        core.compile_model(str(output_dir / "embed.xml"), "CPU")
        for i in range(cfg.num_hidden_layers):
            print(f"  layer_{i} ...")
            core.compile_model(str(output_dir / f"layer_{i}.xml"), "CPU")
        print("  lm_head ...")
        core.compile_model(str(output_dir / "lm_head.xml"), "CPU")
        print("✓ All compile checks passed!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
