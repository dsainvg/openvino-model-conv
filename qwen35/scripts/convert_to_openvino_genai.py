"""Convert Qwen3.5-4B → single stateful OpenVINO model for GenAI / NPU.

This script produces the directory layout expected by openvino_genai.LLMPipeline:

  <output_dir>/
    openvino_model.xml          ← single stateful INT4 model
    openvino_model.bin
    openvino_tokenizer.xml      ← OV tokenizer (from openvino_tokenizers)
    openvino_tokenizer.bin
    openvino_detokenizer.xml    ← OV detokenizer
    openvino_detokenizer.bin
    config.json                 ← HF config (copied)
    tokenizer_config.json       ← HF tokenizer config (copied)
    generation_config.json      ← GenAI generation defaults

Stateful model design
---------------------
1. Trace QwenGenAIWrapper with dummy inputs INCLUDING all state tensors.
2. Call ov.make_stateful() to bind each (state_input_i, state_output_i) pair
   into ReadValue / Assign nodes — the state lives inside the model graph.
3. The public interface shrinks to:
       Inputs : input_ids, attention_mask, position_ids, beam_idx
       Output : logits
4. Apply NNCF INT4_SYM weight compression.
5. Export tokenizer via openvino_tokenizers.

Usage (Kaggle)
--------------
  python scripts/convert_to_openvino_genai.py \\
      --model-dir  /kaggle/working/Qwen3.5-4B \\
      --output     /kaggle/working/ov_genai_qwen35_int4 \\
      --max-seq    2048 \\
      --dtype      bf16

Then in Python (CPU or NPU):
  import openvino_genai as ov_genai
  pipe = ov_genai.LLMPipeline("/kaggle/working/ov_genai_qwen35_int4", "NPU")
  print(pipe.generate("Hello, my name is", max_new_tokens=64))
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# ── Kill OV telemetry ─────────────────────────────────────────────────────────
try:
    import openvino_telemetry as _ot
    _noop = lambda *a, **k: None
    _ot.Telemetry.send_event = _ot.Telemetry.start_session = _noop
    _ot.Telemetry.end_session = _ot.Telemetry.send_error = _noop
    _ot.Telemetry.send_stack_trace = _noop
    try:
        import openvino_telemetry.utils.sender as _s
        _s.TelemetrySender.send = _noop
    except Exception:
        pass
except ImportError:
    pass

import openvino as ov
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
from src.split_inference import QwenGenAIWrapper


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def copy_config_files(src_dir: Path, dst_dir: Path) -> None:
    """Copy tokenizer / config JSON files; skip weight files."""
    skip = {".safetensors", ".bin", ".pt", ".ckpt", ".h5", ".msgpack", ".ot", ".xml"}
    for item in src_dir.iterdir():
        if item.is_file() and item.suffix.lower() not in skip:
            shutil.copy2(item, dst_dir / item.name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Qwen3.5-4B to a single stateful OpenVINO model (GenAI / NPU)."
    )
    p.add_argument("--model-dir", type=Path, required=True,
                   help="HuggingFace weights directory.")
    p.add_argument("--output", type=Path, required=True,
                   help="Output directory (will be created).")
    p.add_argument("--max-seq", type=int, default=2048,
                   help="Static KV-cache length baked into the model (NPU requires static). "
                        "Default: 2048. Use 4096 for longer context (more VRAM).")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16",
                   help="Weight dtype for tracing (default bf16).")
    p.add_argument("--group-size", type=int, default=128,
                   help="NNCF INT4 group size (default 128).")
    p.add_argument("--no-int4", action="store_true",
                   help="Skip INT4 quantization; save as FP16.")
    p.add_argument("--skip-tokenizer", action="store_true",
                   help="Skip openvino_tokenizers export (use if not installed).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Weight loading helpers (full model at once, one layer at a time in memory)
# ─────────────────────────────────────────────────────────────────────────────

def _load_full_model(
    cfg: Qwen35Config,
    weight_map: dict,
    model_dir: Path,
    out_dtype: torch.dtype,
) -> QwenForCausalLM:
    """Build the full QwenForCausalLM and load all weights layer-by-layer."""
    print("  Allocating shell model...")
    model = QwenForCausalLM(cfg)

    # ── Embedding ──────────────────────────────────────────────────────
    embed_key = _find_key(weight_map, "embed_tokens.weight", [
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "model.embed_tokens.weight",
    ])
    model.model.embed_tokens.weight.data = (
        _load_safetensors_weight(weight_map, model_dir, embed_key, out_dtype)
    )
    print(f"  embed_tokens ✓")

    # ── Decoder layers (one at a time to keep RAM low) ─────────────────
    for i in range(cfg.num_hidden_layers):
        load_layer_weights(model.model.layers[i], i, weight_map, model_dir, out_dtype)
        print(f"  layer_{i} ✓", flush=True)
        gc.collect()

    # ── Final norm ─────────────────────────────────────────────────────
    norm_key = _find_key(weight_map, "norm.weight", [
        "model.language_model.norm.weight",
        "language_model.model.norm.weight",
        "model.norm.weight",
    ])
    model.model.norm.weight.data = (
        _load_safetensors_weight(weight_map, model_dir, norm_key, out_dtype)
    )

    # ── LM Head (handle tie_word_embeddings) ───────────────────────────
    try:
        lm_key = _find_key(weight_map, "lm_head.weight", [
            "lm_head.weight",
            "model.language_model.lm_head.weight",
            "language_model.lm_head.weight",
        ])
        lm_prefix = lm_key.removesuffix(".weight")
        lm_weight  = load_linear_weight(weight_map, model_dir, lm_prefix, out_dtype)
    except KeyError:
        print("  (lm_head not in checkpoint — tie_word_embeddings, reusing embed weight)")
        lm_weight = model.model.embed_tokens.weight.data.clone()

    if lm_weight.shape != model.lm_head.weight.shape:
        model.lm_head = torch.nn.Linear(cfg.hidden_size, lm_weight.shape[0], bias=False)
    model.lm_head.weight.data = lm_weight
    print("  lm_head ✓")

    return model.eval().to(out_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Dummy state builder
# ─────────────────────────────────────────────────────────────────────────────

def _make_dummy_states(
    cfg: Qwen35Config,
    max_seq: int,
    dtype: torch.dtype,
    batch: int = 1,
) -> list[torch.Tensor]:
    """Return flat dummy state tensors in the same order as QwenGenAIWrapper."""
    full_indices   = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    linear_indices = [i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"]

    states: list[torch.Tensor] = []

    # k_caches
    for _ in full_indices:
        states.append(torch.zeros(batch, cfg.num_key_value_heads, max_seq, cfg.head_dim, dtype=dtype))
    # v_caches
    for _ in full_indices:
        states.append(torch.zeros(batch, cfg.num_key_value_heads, max_seq, cfg.head_dim, dtype=dtype))
    # conv_states
    for _ in linear_indices:
        states.append(torch.zeros(batch, cfg.linear_conv_dim, cfg.linear_conv_kernel_dim, dtype=dtype))
    # rec_states
    for _ in linear_indices:
        states.append(torch.zeros(
            batch, cfg.linear_num_value_heads,
            cfg.linear_key_head_dim, cfg.linear_value_head_dim, dtype=dtype,
        ))

    return states


# ─────────────────────────────────────────────────────────────────────────────
# Stateful wiring
# ─────────────────────────────────────────────────────────────────────────────

def _make_stateful(
    ov_model: ov.Model,
    num_full: int,
    num_linear: int,
) -> ov.Model:
    """Bind state input ports to state output ports with MakeStateful.

    Input port order  (after input_ids, attention_mask, position_ids, beam_idx):
        k_cache_0..N-1, v_cache_0..N-1, conv_0..M-1, rec_0..M-1
    Output port order (after logits):
        k_cache_0..N-1, v_cache_0..N-1, conv_0..M-1, rec_0..M-1

    where N = num_full, M = num_linear.
    """
    try:
        from openvino.runtime.passes import Manager, MakeStateful
    except ImportError:
        from openvino.passes import Manager, MakeStateful

    inputs  = ov_model.inputs
    outputs = ov_model.outputs

    # Fixed inputs: input_ids(0), attention_mask(1), position_ids(2), beam_idx(3)
    # Fixed outputs: logits(0)
    STATE_INPUT_OFFSET  = 4
    STATE_OUTPUT_OFFSET = 1

    n_states = 2 * num_full + 2 * num_linear
    tensor_names = {}
    for k in range(n_states):
        inp_port = inputs[STATE_INPUT_OFFSET + k]
        out_port = outputs[STATE_OUTPUT_OFFSET + k]
        tensor_names[inp_port.any_name] = out_port.any_name

    manager = Manager()
    manager.register_pass(MakeStateful(tensor_names))
    manager.run_passes(ov_model)
    return ov_model


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer export
# ─────────────────────────────────────────────────────────────────────────────

def export_tokenizer(model_dir: Path, output_dir: Path) -> None:
    """Convert HF tokenizer → openvino_tokenizer.xml/bin."""
    try:
        from openvino_tokenizers import convert_tokenizer
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"  ⚠ Cannot export OV tokenizer: {e}")
        print("    Install with: pip install openvino-tokenizers transformers")
        return

    print("  Loading HF tokenizer...")
    hf_tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)

    print("  Converting to OV tokenizer...")
    ov_tok, ov_detok = convert_tokenizer(hf_tok, with_detokenizer=True)

    ov.save_model(ov_tok,   str(output_dir / "openvino_tokenizer.xml"))
    ov.save_model(ov_detok, str(output_dir / "openvino_detokenizer.xml"))
    print("  openvino_tokenizer.xml ✓")
    print("  openvino_detokenizer.xml ✓")


# ─────────────────────────────────────────────────────────────────────────────
# Generation config
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args      = parse_args()
    model_dir = args.model_dir.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    out_dtype  = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    use_int4   = not args.no_int4
    max_seq    = args.max_seq
    group_size = args.group_size

    print(f"Model dir   : {model_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Max seq len : {max_seq} (static KV-cache)")
    print(f"Weight dtype: {args.dtype}")
    print(f"Quantization: {'INT4_SYM  group_size=' + str(group_size) if use_int4 else 'FP16 (--no-int4)'}")

    cfg        = Qwen35Config.from_pretrained_dir(model_dir)
    weight_map = build_shard_index(model_dir)

    # ─────────────────────────────────────────────────────────────────
    # 1. Load full model
    # ─────────────────────────────────────────────────────────────────
    print("\n[1/5] Loading model weights (layer-by-layer to limit RAM)...")
    t0 = time.time()
    full_model = _load_full_model(cfg, weight_map, model_dir, out_dtype)
    print(f"  Done ({time.time()-t0:.1f}s)")

    # ─────────────────────────────────────────────────────────────────
    # 2. Build wrapper + example inputs
    # ─────────────────────────────────────────────────────────────────
    print("\n[2/5] Building stateful wrapper + example inputs...")
    wrapper = QwenGenAIWrapper(full_model).eval()

    num_full   = wrapper.num_full
    num_linear = wrapper.num_linear

    dummy_states = _make_dummy_states(cfg, max_seq, out_dtype, batch=1)

    example_inputs = (
        torch.tensor([[1]], dtype=torch.long),           # input_ids  [1,1]
        torch.tensor([[1]], dtype=torch.long),           # attention_mask [1,1]
        torch.tensor([[0]], dtype=torch.long),           # position_ids [1,1]
        torch.tensor([0],   dtype=torch.long),           # beam_idx [1]
        *dummy_states,
    )
    print(f"  num_full_layers   = {num_full}")
    print(f"  num_linear_layers = {num_linear}")
    print(f"  total state tensors = {len(dummy_states)}")

    # ─────────────────────────────────────────────────────────────────
    # 3. Trace to OV IR
    # ─────────────────────────────────────────────────────────────────
    print("\n[3/5] Tracing to OpenVINO IR (this takes a few minutes)...")
    t0 = time.time()
    ov_model = ov.convert_model(wrapper, example_input=example_inputs)
    print(f"  Trace complete ({time.time()-t0:.1f}s)")

    # ─────────────────────────────────────────────────────────────────
    # 4. Make stateful (bind state in → state out as ReadValue/Assign)
    # ─────────────────────────────────────────────────────────────────
    print("\n[4/5] Making stateful (binding KV/recurrent state)...")
    ov_model = _make_stateful(ov_model, num_full, num_linear)
    print("  Stateful wiring done.")
    print(f"  Public inputs  : {[i.any_name for i in ov_model.inputs]}")
    print(f"  Public outputs : {[o.any_name for o in ov_model.outputs]}")

    # Free PyTorch model memory before INT4 (NNCF needs headroom)
    del full_model, wrapper
    gc.collect()

    # ─────────────────────────────────────────────────────────────────
    # 5a. INT4 compression
    # ─────────────────────────────────────────────────────────────────
    print("\n[5/5] Saving model...")
    if use_int4:
        print(f"  Applying NNCF INT4_SYM (group_size={group_size})...")
        t0 = time.time()
        ov_model = compress_weights(
            ov_model,
            mode=CompressWeightsMode.INT4_SYM,
            group_size=group_size,
            ratio=1.0,
        )
        print(f"  INT4 compression done ({time.time()-t0:.1f}s)")

    xml_path = str(output_dir / "openvino_model.xml")
    ov.save_model(ov_model, xml_path, compress_to_fp16=True)
    del ov_model
    gc.collect()

    # File size summary
    bin_path = output_dir / "openvino_model.bin"
    if bin_path.exists():
        mb = bin_path.stat().st_size / 1e6
        print(f"  openvino_model.xml ✓  ({mb:.0f} MB)")
    else:
        print("  openvino_model.xml ✓")

    # ─────────────────────────────────────────────────────────────────
    # 5b. Tokenizer export
    # ─────────────────────────────────────────────────────────────────
    if not args.skip_tokenizer:
        print("\n  Exporting tokenizer...")
        export_tokenizer(model_dir, output_dir)
    else:
        print("\n  Skipping tokenizer export (--skip-tokenizer).")

    # ─────────────────────────────────────────────────────────────────
    # 5c. Generation config + config files
    # ─────────────────────────────────────────────────────────────────
    write_generation_config(cfg, output_dir)
    copy_config_files(model_dir, output_dir)

    print(f"\n✓ GenAI-ready model saved to: {output_dir}")
    print()
    print("  Run on CPU:")
    print("    import openvino_genai as ov_genai")
    print(f"    pipe = ov_genai.LLMPipeline('{output_dir}', 'CPU')")
    print("    print(pipe.generate('Hello', max_new_tokens=64))")
    print()
    print("  Run on NPU:")
    print(f"    pipe = ov_genai.LLMPipeline('{output_dir}', 'NPU')")

    return 0


if __name__ == "__main__":
    sys.exit(main())
