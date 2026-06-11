"""Convert Qwen3.5-4B to OpenVINO IR via ov.convert_model.

Mirrors qwen36/scripts/convert_toy.py but works on the real checkpoint:
  1. Read config.json → build QwenForCausalLM
  2. Load BF16 weights from safetensors shards (src/load_weights.py)
  3. Wrap forward() in FlatWrapper so every state tensor is a distinct OV port
  4. ov.convert_model with dynamic shapes
  5. ov.save_model (compress_to_fp16=True)
  6. Reload + compile on CPU, run a sanity forward to confirm finite outputs

Usage:
    python scripts/convert_to_openvino.py --model-dir /path/to/Qwen3.5-4B
    python scripts/convert_to_openvino.py --model-dir /path/to/Qwen3.5-4B \\
        --output /path/to/ov_ir_qwen35_4b --compile-check
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import openvino as ov

from src.configuration import Qwen35Config
from src.load_weights import build_shard_index, load_global_weights, load_layer_weights
from src.modeling import QwenForCausalLM


# ---------------------------------------------------------------------------
# OV trace wrapper
# ---------------------------------------------------------------------------


class FlatWrapper(nn.Module):
    """Flatten list-valued state args into positional tensors so each becomes
    a distinct named OV graph input/output — same pattern as qwen36's wrapper.
    """

    def __init__(self, model: QwenForCausalLM):
        super().__init__()
        self.model   = model
        self.n_full  = model.model.num_full_layers
        self.n_lin   = model.model.num_linear_layers

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # args layout:
        #   input_ids, position_ids,
        #   k_cache_0 .. k_cache_{n_full-1},
        #   v_cache_0 .. v_cache_{n_full-1},
        #   conv_state_0 .. conv_state_{n_lin-1},
        #   rec_state_0  .. rec_state_{n_lin-1}
        input_ids, position_ids = args[0], args[1]
        i = 2
        k_caches    = list(args[i : i + self.n_full]);  i += self.n_full
        v_caches    = list(args[i : i + self.n_full]);  i += self.n_full
        conv_states = list(args[i : i + self.n_lin]);   i += self.n_lin
        rec_states  = list(args[i : i + self.n_lin]);   i += self.n_lin

        logits, k_out, v_out, conv_out, rec_out = self.model(
            input_ids, position_ids, k_caches, v_caches, conv_states, rec_states
        )
        return (logits, *k_out, *v_out, *conv_out, *rec_out)


def build_example_inputs(model: QwenForCausalLM, max_seq: int = 8):
    state = model.empty_state(batch=1, max_seq=max_seq)
    return (
        torch.tensor([[1]], dtype=torch.long),
        torch.tensor([[0]], dtype=torch.long),
        *state["k_caches"],
        *state["v_caches"],
        *state["conv_states"],
        *state["rec_states"],
    )


def copy_non_weights_files(src_dir: Path, dst_dir: Path) -> None:
    import shutil
    # List of extensions to ignore (weights/tensors)
    ignore_suffixes = {".safetensors", ".bin", ".pt", ".ckpt", ".h5", ".msgpack", ".ot"}
    for item in src_dir.iterdir():
        if item.is_file():
            if item.suffix.lower() not in ignore_suffixes:
                dst_file = dst_dir / item.name
                print(f"  copying {item.name} ...")
                shutil.copy2(item, dst_file)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Qwen3.5-4B to OpenVINO IR.")
    p.add_argument("--model-dir", type=Path, required=True,
                   help="Path to the snapshot_download output directory.")
    p.add_argument("--output", type=Path, default=None,
                   help="Output directory for IR files. "
                        "Defaults to <repo_root>/ov_ir_qwen35_4b.")
    p.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16",
                   help="Dtype to load weights in before tracing (default bf16).")
    p.add_argument("--compile-check", action="store_true",
                   help="After saving, reload IR and run a forward to check for errors.")
    p.add_argument("--layers", type=int, default=None,
                   help="Load only the first N layers (rest keep random init). "
                        "Useful for debugging a partial model.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    repo_root  = Path(__file__).resolve().parents[1]
    output_dir = args.output.resolve() if args.output else (repo_root.parent / "ov_ir_qwen35_4b")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir  = args.model_dir.resolve()

    out_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    print(f"model dir   : {model_dir}")
    print(f"output dir  : {output_dir}")
    print(f"weight dtype: {args.dtype}")

    # 1. Build model from real config
    print("\nBuilding model from config ...")
    cfg   = Qwen35Config.from_pretrained_dir(model_dir)
    model = QwenForCausalLM(cfg)
    model.eval()
    print(f"  layers      : {cfg.num_hidden_layers}")
    print(f"  hidden_size : {cfg.hidden_size}")
    print(f"  vocab_size  : {cfg.vocab_size}")
    print(f"  full-attn   : {model.model.num_full_layers} layers")
    print(f"  linear-attn : {model.model.num_linear_layers} layers")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters  : {total_params/1e9:.2f}B")

    # 2. Load weights
    print("\nLoading weights ...")
    weight_map = build_shard_index(model_dir)
    load_global_weights(model, weight_map, model_dir, out_dtype=out_dtype)
    n_layers = args.layers if args.layers is not None else cfg.num_hidden_layers
    for i in range(n_layers):
        print(f"  layer {i+1}/{n_layers}", end="\r")
        load_layer_weights(model.model.layers[i], i, weight_map, model_dir, out_dtype=out_dtype)
    print(f"\n  Loaded {n_layers} layer(s).")

    # 3. Wrap and build example inputs
    wrapper        = FlatWrapper(model).eval()
    example_inputs = build_example_inputs(model, max_seq=8)

    # 4. PyTorch reference forward
    print("\nRunning PyTorch reference forward ...")
    with torch.no_grad():
        ref_outputs = wrapper(*example_inputs)
    ref_logits = ref_outputs[0]
    print(f"  logits shape : {tuple(ref_logits.shape)}")
    print(f"  logits finite: {torch.isfinite(ref_logits).all().item()}")

    # 5. ov.convert_model
    print("\nov.convert_model ...")
    t0       = time.time()
    ov_model = ov.convert_model(wrapper, example_input=example_inputs)
    print(f"  OK ({time.time()-t0:.1f}s)")
    ov_model.outputs[0].set_names({"logits"})

    # 6. Save IR
    ir_xml = output_dir / "openvino_model.xml"
    ov.save_model(ov_model, str(ir_xml), compress_to_fp16=True)
    ir_bin = ir_xml.with_suffix(".bin")
    print(f"\nSaved IR:")
    print(f"  {ir_xml}  ({ir_xml.stat().st_size/1e6:.1f} MB)")
    print(f"  {ir_bin}  ({ir_bin.stat().st_size/1e6:.1f} MB)")

    # Copy tokenizer and config files
    print("\nCopying tokenizer and config files ...")
    copy_non_weights_files(model_dir, output_dir)

    # 7. Optional compile check + numerical sanity
    if args.compile_check:
        print("\nCompile-check: reloading IR on CPU ...")
        core     = ov.Core()
        print(f"  available devices: {core.available_devices}")
        compiled = core.compile_model(str(ir_xml), "CPU")
        print(f"  compiled — {len(compiled.inputs)} inputs / {len(compiled.outputs)} outputs")
        ov_result = compiled([t.numpy() for t in example_inputs])
        ov_logits = torch.from_numpy(ov_result[0])
        diff      = (ref_logits.float() - ov_logits.float()).abs()
        pt_next   = ref_logits[0, -1].float().argmax().item()
        ov_next   = ov_logits[0, -1].float().argmax().item()
        print(f"  abs diff  max={diff.max().item():.3e}  mean={diff.mean().item():.3e}")
        print(f"  greedy next-token  PT={pt_next}  OV={ov_next}  match={pt_next == ov_next}")

    print("\nQwen3.5-4B OpenVINO conversion complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
