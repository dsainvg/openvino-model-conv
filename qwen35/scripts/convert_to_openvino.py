"""Convert Qwen3.5-4B to OpenVINO IR in split-IR format (layer-by-layer).

This script:
  1. Reads config.json → build Qwen3.5 components.
  2. Loads and exports Embedding module to embed.xml.
  3. Loads and exports each Decoder Layer sequentially (freeing RAM immediately) to layer_N.xml.
  4. Loads and exports LM Head module to lm_head.xml.
This avoids loading the entire model into RAM at once, preventing memory crashes.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Disable telemetry
try:
    import openvino_telemetry
    def dummy_method(*args, **kwargs):
        pass
    openvino_telemetry.Telemetry.send_event = dummy_method
    openvino_telemetry.Telemetry.start_session = dummy_method
    openvino_telemetry.Telemetry.end_session = dummy_method
    openvino_telemetry.Telemetry.send_error = dummy_method
    openvino_telemetry.Telemetry.send_stack_trace = dummy_method
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
from src.split_inference import QwenEmbedWrapper, QwenLayerWrapper, QwenLMHeadWrapper


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
                   help="Perform a sequential compile-check to verify the IR files.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    output_dir = args.output.resolve() if args.output else (REPO.parent / "ov_ir_qwen35_4b")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir  = args.model_dir.resolve()

    out_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(f"Model dir   : {model_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Weight dtype: {args.dtype}")

    # Load configuration
    cfg = Qwen35Config.from_pretrained_dir(model_dir)
    weight_map = build_shard_index(model_dir)

    # 1. Convert Embedding
    print("\n[1/3] Converting Embedding layer...")
    dummy_model = QwenForCausalLM(cfg)
    embed_candidates = [
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "model.embed_tokens.weight",
    ]
    embed_key = _find_key(weight_map, "embed_tokens.weight", embed_candidates)
    dummy_model.model.embed_tokens.weight.data.copy_(
        _load_safetensors_weight(weight_map, model_dir, embed_key, out_dtype)
    )
    embed_wrapper = QwenEmbedWrapper(dummy_model).eval()
    
    onnx_embed_path = output_dir / "embed.onnx"
    torch.onnx.export(
        embed_wrapper,
        (torch.tensor([[1]], dtype=torch.long),),
        str(onnx_embed_path),
        input_names=["input_ids"],
        output_names=["x"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "x": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=18,
    )
    ov_embed = ov.convert_model(str(onnx_embed_path))
    ov.save_model(ov_embed, str(output_dir / "embed.xml"), compress_to_fp16=True)
    onnx_embed_path.unlink(missing_ok=True)
    
    del dummy_model, embed_wrapper, ov_embed
    gc.collect()

    # 2. Convert Decoder Layers
    print("\n[2/3] Converting Decoder layers sequentially...")
    for i in range(cfg.num_hidden_layers):
        print(f"  Layer {i+1}/{cfg.num_hidden_layers} ...")
        layer = QwenDecoderLayer(cfg, layer_idx=i).eval()
        load_layer_weights(layer, i, weight_map, model_dir, out_dtype=out_dtype)
        layer_wrapper = QwenLayerWrapper(layer).eval()

        onnx_layer_path = output_dir / f"layer_{i}.onnx"
        if layer.layer_type == "full_attention":
            x = torch.randn(1, 1, cfg.hidden_size)
            cos = torch.randn(1, 1, cfg.partial_rotary_dim)
            sin = torch.randn(1, 1, cfg.partial_rotary_dim)
            k = torch.zeros(1, cfg.num_key_value_heads, 8, cfg.head_dim)
            v = torch.zeros_like(k)
            write_pos = torch.tensor(0, dtype=torch.long)
            example_inputs = (x, cos, sin, k, v, write_pos)

            input_names = ["x", "cos", "sin", "k_cache", "v_cache", "write_pos"]
            output_names = ["x_out", "k_out", "v_out"]
            dynamic_axes = {
                "x": {0: "batch_size"},
                "cos": {0: "batch_size"},
                "sin": {0: "batch_size"},
                "k_cache": {0: "batch_size", 2: "seq_len"},
                "v_cache": {0: "batch_size", 2: "seq_len"},
                "x_out": {0: "batch_size"},
                "k_out": {0: "batch_size", 2: "seq_len"},
                "v_out": {0: "batch_size", 2: "seq_len"},
            }
        else:
            x = torch.randn(1, 1, cfg.hidden_size)
            conv = torch.zeros(1, cfg.linear_conv_dim, cfg.linear_conv_kernel_dim)
            rec = torch.zeros(1, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim)
            example_inputs = (x, conv, rec)

            input_names = ["x", "conv_state", "recurrent_state"]
            output_names = ["x_out", "conv_out", "recurrent_out"]
            dynamic_axes = {
                "x": {0: "batch_size"},
                "conv_state": {0: "batch_size"},
                "recurrent_state": {0: "batch_size"},
                "x_out": {0: "batch_size"},
                "conv_out": {0: "batch_size"},
                "recurrent_out": {0: "batch_size"},
            }

        torch.onnx.export(
            layer_wrapper,
            example_inputs,
            str(onnx_layer_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=18,
        )
        ov_layer = ov.convert_model(str(onnx_layer_path))
        ov.save_model(ov_layer, str(output_dir / f"layer_{i}.xml"), compress_to_fp16=True)
        onnx_layer_path.unlink(missing_ok=True)

        del layer, layer_wrapper, ov_layer
        gc.collect()

    # 3. Convert LM Head & final norm
    print("\n[3/3] Converting LM Head and norm...")
    dummy_model = QwenForCausalLM(cfg)
    
    norm_candidates = [
        "model.language_model.norm.weight",
        "language_model.model.norm.weight",
        "model.norm.weight",
    ]
    lm_head_candidates = [
        "lm_head.weight",
        "model.language_model.lm_head.weight",
        "language_model.lm_head.weight",
        "lm_head.qweight",
        "model.language_model.lm_head.qweight",
        "language_model.lm_head.qweight",
    ]

    norm_key = _find_key(weight_map, "norm.weight", norm_candidates)
    try:
        lm_head_key = _find_key(weight_map, "lm_head.weight", lm_head_candidates)
    except KeyError:
        lm_head_key = _find_key(weight_map, "lm_head.qweight", lm_head_candidates)

    dummy_model.model.norm.weight.data.copy_(
        _load_safetensors_weight(weight_map, model_dir, norm_key, out_dtype)
    )

    if lm_head_key.endswith(".qweight"):
        lm_head_prefix = lm_head_key[:-8]
    elif lm_head_key.endswith(".weight"):
        lm_head_prefix = lm_head_key[:-7]
    else:
        lm_head_prefix = lm_head_key

    dummy_model.lm_head.weight.data.copy_(
        load_linear_weight(weight_map, model_dir, lm_head_prefix, out_dtype)
    )

    lm_head_wrapper = QwenLMHeadWrapper(dummy_model).eval()
    onnx_lm_path = output_dir / "lm_head.onnx"
    torch.onnx.export(
        lm_head_wrapper,
        (torch.randn(1, 1, cfg.hidden_size),),
        str(onnx_lm_path),
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={
            "x": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        },
        opset_version=18,
    )
    ov_lm = ov.convert_model(str(onnx_lm_path))
    ov.save_model(ov_lm, str(output_dir / "lm_head.xml"), compress_to_fp16=True)
    onnx_lm_path.unlink(missing_ok=True)

    del dummy_model, lm_head_wrapper, ov_lm
    gc.collect()

    # Copy tokenizer and config
    copy_non_weights_files(model_dir, output_dir)
    print("\nQwen3.5-4B split-IR conversion complete.")

    # Low-memory sequential compile check
    if args.compile_check:
        print("\nCompile-check: sequentially compiling split IRs on CPU...")
        core = ov.Core()
        print("  Compiling embed...")
        core.compile_model(str(output_dir / "embed.xml"), "CPU")
        for i in range(cfg.num_hidden_layers):
            print(f"  Compiling layer {i}...")
            core.compile_model(str(output_dir / f"layer_{i}.xml"), "CPU")
        print("  Compiling lm_head...")
        core.compile_model(str(output_dir / "lm_head.xml"), "CPU")
        print("✓ Sequential compile check successful!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
