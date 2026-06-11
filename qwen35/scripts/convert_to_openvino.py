"""Convert Qwen3.5-4B to OpenVINO INT4 split-IR format (layer-by-layer).

Strategy:
  1. Feed each PyTorch wrapper directly to ov.convert_model() — no ONNX step.
  2. Apply NNCF INT4 weight compression (nncf.compress_weights) to every IR.
  3. Save each IR immediately and free RAM before loading the next layer.

INT4 quantization uses AWQ-style asymmetric per-group quantization:
  - mode  : INT4_ASYM  (best quality for LLMs)
  - group_size: 64  (balances quality vs size; 128 is faster but slightly lower quality)
  - ratio : 1.0  (compress ALL Linear layers to INT4)
  - Embeddings and norms stay in fp16 (handled by compress_to_fp16 at save time).

Output layout:
  <output_dir>/embed.xml        -- embedding lookup (fp16 weights)
  <output_dir>/layer_N.xml      -- decoder layers  (INT4 linear weights)
  <output_dir>/lm_head.xml      -- final norm + lm_head (INT4 projection)
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
    openvino_telemetry.Telemetry.send_event       = _noop
    openvino_telemetry.Telemetry.start_session    = _noop
    openvino_telemetry.Telemetry.end_session      = _noop
    openvino_telemetry.Telemetry.send_error       = _noop
    openvino_telemetry.Telemetry.send_stack_trace = _noop
    try:
        import openvino_telemetry.utils.sender
        openvino_telemetry.utils.sender.TelemetrySender.send = _noop
    except Exception:
        pass
except ImportError:
    pass

import openvino as ov
import nncf
from nncf import compress_weights, CompressWeightsMode

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
# INT4 compression config
# ─────────────────────────────────────────────────────────────────────────────
INT4_MODE       = CompressWeightsMode.INT4_SYM  # Symmetric mode requested by user
INT4_GROUP_SIZE = 128   # User requested group size 128
INT4_RATIO      = 1.0   # compress ALL linear weights to INT4


def _save_int4(ov_model: ov.Model, xml_path: str) -> None:
    """Apply NNCF INT4 weight compression and save the IR."""
    compressed = compress_weights(
        ov_model,
        mode=INT4_MODE,
        group_size=INT4_GROUP_SIZE,
        ratio=INT4_RATIO,
    )
    ov.save_model(compressed, xml_path, compress_to_fp16=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def copy_non_weights_files(src_dir: Path, dst_dir: Path) -> None:
    import shutil
    ignore_suffixes = {".safetensors", ".bin", ".pt", ".ckpt", ".h5", ".msgpack", ".ot", ".xml"}
    for item in src_dir.iterdir():
        if item.is_file() and item.suffix.lower() not in ignore_suffixes:
            shutil.copy2(item, dst_dir / item.name)


def export_tokenizer(model_dir: Path, output_dir: Path) -> None:
    """Convert HF tokenizer → openvino_tokenizer.xml/bin."""
    try:
        from openvino_tokenizers import convert_tokenizer
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"  ⚠ Cannot export OV tokenizer: {e}")
        return

    print("  Converting to OV tokenizer...")
    hf_tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    ov_tok, ov_detok = convert_tokenizer(hf_tok, with_detokenizer=True)

    ov.save_model(ov_tok,   str(output_dir / "openvino_tokenizer.xml"))
    ov.save_model(ov_detok, str(output_dir / "openvino_detokenizer.xml"))
    print("  openvino_tokenizer.xml ✓")
    print("  openvino_detokenizer.xml ✓")


def write_generation_config(cfg: Qwen35Config, output_dir: Path) -> None:
    """Write generation_config.json required by openvino_genai."""
    gen_cfg = {
        "bos_token_id": cfg.bos_token_id,
        "eos_token_id": cfg.eos_token_id,
        "pad_token_id": cfg.pad_token_id,
        "max_new_tokens": 512,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.0,
        "transformers_version": "4.57.0.dev0",
    }
    with open(output_dir / "generation_config.json", "w", encoding="utf-8") as f:
        json.dump(gen_cfg, f, indent=2)
    print("  generation_config.json ✓")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Qwen3.5-4B to INT4 split-IR OpenVINO."
    )
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Path to the HuggingFace weights directory.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output directory for split IR files.")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16",
                   help="Dtype for tracing (default bf16).")
    p.add_argument("--group-size", type=int, default=INT4_GROUP_SIZE,
                   help=f"NNCF INT4 group size (default {INT4_GROUP_SIZE}).")
    p.add_argument("--no-int4", action="store_true",
                   help="Skip INT4 quantization; save as FP16 instead.")
    p.add_argument("--compile-check", action="store_true",
                   help="Sequentially compile each IR on CPU to verify correctness.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    output_dir = args.output.resolve() if args.output else (REPO.parent / "ov_ir_qwen35_4b_int4")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir  = args.model_dir.resolve()

    out_dtype  = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    use_int4   = not args.no_int4
    group_size = args.group_size

    print(f"Model dir   : {model_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Weight dtype: {args.dtype}")
    print(f"Quantization: {'INT4_SYM  group_size=' + str(group_size) if use_int4 else 'FP16 (--no-int4)'}")

    cfg        = Qwen35Config.from_pretrained_dir(model_dir)
    weight_map = build_shard_index(model_dir)

    def save_ir(ov_model: ov.Model, xml_path: str) -> None:
        if use_int4:
            compressed = compress_weights(
                ov_model,
                mode=INT4_MODE,
                group_size=group_size,
                ratio=INT4_RATIO,
            )
            ov.save_model(compressed, xml_path, compress_to_fp16=True)
        else:
            ov.save_model(ov_model, xml_path, compress_to_fp16=True)

    # ──────────────────────────────────────────────────────────────
    # 1. Embedding  (kept FP16 — embeddings don't benefit from INT4)
    # ──────────────────────────────────────────────────────────────
    print("\n[1/3] Converting Embedding (FP16)...")
    shell = QwenForCausalLM(cfg)

    embed_key = _find_key(weight_map, "embed_tokens.weight", [
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "model.embed_tokens.weight",
    ])
    shell.model.embed_tokens.weight.data = (
        _load_safetensors_weight(weight_map, model_dir, embed_key, out_dtype)
    )

    embed_wrapper = QwenEmbedWrapper(shell).eval().to(out_dtype)
    example_ids   = torch.tensor([[1, 2, 3]], dtype=torch.long)

    ov_embed = ov.convert_model(embed_wrapper, example_input=(example_ids,))
    # Embedding is always FP16 — INT4 on a lookup table saves nothing
    ov.save_model(ov_embed, str(output_dir / "embed_tokens.xml"), compress_to_fp16=True)
    del shell, embed_wrapper, ov_embed
    gc.collect()
    print("  embed_tokens.xml ✓ (fp16)")

    # ──────────────────────────────────────────────────────────────
    # 2. Decoder Layers  (INT4 — the big savings are here)
    # ──────────────────────────────────────────────────────────────
    print(f"\n[2/3] Converting {cfg.num_hidden_layers} decoder layers (INT4)...")
    for i in range(cfg.num_hidden_layers):
        t0 = time.time()
        lt = cfg.layer_types[i]
        print(f"  [{i+1}/{cfg.num_hidden_layers}] layer_{i}  ({lt}) ...", flush=True)

        layer = QwenDecoderLayer(cfg, layer_idx=i).eval()
        load_layer_weights(layer, i, weight_map, model_dir, out_dtype=out_dtype)

        if lt == "full_attention":
            wrapper   = QwenLayerFullWrapper(layer).eval().to(out_dtype)
            x         = torch.randn(1, 1, cfg.hidden_size,         dtype=out_dtype)
            cos        = torch.randn(1, 1, cfg.partial_rotary_dim,  dtype=out_dtype)
            sin        = torch.randn(1, 1, cfg.partial_rotary_dim,  dtype=out_dtype)
            k_cache    = torch.zeros(1, cfg.num_key_value_heads, 8, cfg.head_dim, dtype=out_dtype)
            v_cache    = torch.zeros_like(k_cache)
            write_pos  = torch.tensor(0, dtype=torch.long)
            example_in = (x, cos, sin, k_cache, v_cache, write_pos)
        else:  # linear_attention
            wrapper   = QwenLayerLinearWrapper(layer).eval().to(out_dtype)
            x         = torch.randn(1, 1, cfg.hidden_size,                                      dtype=out_dtype)
            conv      = torch.zeros(1, cfg.linear_conv_dim, cfg.linear_conv_kernel_dim,         dtype=out_dtype)
            rec       = torch.zeros(1, cfg.linear_num_value_heads,
                                    cfg.linear_key_head_dim, cfg.linear_value_head_dim,         dtype=out_dtype)
            example_in = (x, conv, rec)

        ov_layer = ov.convert_model(wrapper, example_input=example_in)
        save_ir(ov_layer, str(output_dir / f"layer_{i}.xml"))

        del layer, wrapper, ov_layer
        gc.collect()
        q_tag = "INT4" if use_int4 else "fp16"
        print(f"    layer_{i}.xml ✓ ({q_tag}, {time.time()-t0:.1f}s)")

    # ──────────────────────────────────────────────────────────────
    # 3. LM Head + final norm (INT4)
    # ──────────────────────────────────────────────────────────────
    print("\n[3/3] Converting LM Head + final norm (INT4)...")
    shell = QwenForCausalLM(cfg)

    norm_key = _find_key(weight_map, "norm.weight", [
        "model.language_model.norm.weight",
        "language_model.model.norm.weight",
        "model.norm.weight",
    ])
    shell.model.norm.weight.data = (
        _load_safetensors_weight(weight_map, model_dir, norm_key, out_dtype)
    )

    # Handle tie_word_embeddings=True: no separate lm_head weight in checkpoint
    try:
        lm_key = _find_key(weight_map, "lm_head.weight", [
            "lm_head.weight",
            "model.language_model.lm_head.weight",
            "language_model.lm_head.weight",
            "lm_head.qweight",
            "model.language_model.lm_head.qweight",
            "language_model.lm_head.qweight",
        ])
        lm_prefix = lm_key.removesuffix(".qweight").removesuffix(".weight")
        lm_weight = load_linear_weight(weight_map, model_dir, lm_prefix, out_dtype)
    except KeyError:
        print("  (lm_head.weight not in checkpoint — tie_word_embeddings, reusing embed weight)")
        embed_key = _find_key(weight_map, "embed_tokens.weight", [
            "model.language_model.embed_tokens.weight",
            "language_model.model.embed_tokens.weight",
            "model.embed_tokens.weight",
        ])
        lm_weight = _load_safetensors_weight(weight_map, model_dir, embed_key, out_dtype)

    # Resize lm_head if vocab size changed (e.g. 248320 vs default)
    if lm_weight.shape != shell.lm_head.weight.shape:
        shell.lm_head = torch.nn.Linear(cfg.hidden_size, lm_weight.shape[0], bias=False)
    shell.lm_head.weight.data = lm_weight

    lm_wrapper = QwenLMHeadWrapper(shell).eval().to(out_dtype)
    example_x  = torch.randn(1, 3, cfg.hidden_size, dtype=out_dtype)

    ov_lm = ov.convert_model(lm_wrapper, example_input=(example_x,))
    save_ir(ov_lm, str(output_dir / "lm_head.xml"))
    del shell, lm_wrapper, ov_lm
    gc.collect()
    print(f"  lm_head.xml ✓ ({'INT4' if use_int4 else 'fp16'})")

    # Copy tokenizer / config alongside the IR
    copy_non_weights_files(model_dir, output_dir)
    export_tokenizer(model_dir, output_dir)
    write_generation_config(cfg, output_dir)

    # Summary
    xmls = sorted(output_dir.glob("*.xml"))
    total_mb = sum((output_dir / x.stem).with_suffix(".bin").stat().st_size
                   for x in xmls
                   if (output_dir / x.stem).with_suffix(".bin").exists()) / 1e6
    print(f"\n✓ INT4 split-IR conversion complete → {output_dir}")
    print(f"  {len(xmls)} IR files, ~{total_mb:.0f} MB total weights")

    # ──────────────────────────────────────────────────────────────
    # Optional compile-check
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
