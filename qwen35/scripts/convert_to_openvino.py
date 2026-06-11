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

# Programmatically disable OpenVINO telemetry to avoid Windows/Python 3.13 thread-safety crashes in ssl/socket
try:
    import openvino_telemetry
    def dummy_method(*args, **kwargs):
        pass
    openvino_telemetry.Telemetry.send_event = dummy_method
    openvino_telemetry.Telemetry.start_session = dummy_method
    openvino_telemetry.Telemetry.end_session = dummy_method
    openvino_telemetry.Telemetry.send_error = dummy_method
    openvino_telemetry.Telemetry.send_stack_trace = dummy_method
    
    import openvino_telemetry.utils.sender
    openvino_telemetry.utils.sender.TelemetrySender.send = dummy_method
except ImportError:
    pass

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

        # Dynamically compile forward method with explicit arguments (no *args list unpacking)
        # to avoid C++ segmentation faults in OpenVINO's JIT frontend.
        args_list = ["self", "input_ids", "position_ids"]
        for i in range(self.n_full):
            args_list.append(f"k_cache_{i}")
        for i in range(self.n_full):
            args_list.append(f"v_cache_{i}")
        for i in range(self.n_lin):
            args_list.append(f"conv_state_{i}")
        for i in range(self.n_lin):
            args_list.append(f"rec_state_{i}")
            
        args_str = ", ".join(args_list)
        
        body = []
        if self.n_full > 0:
            body.append(f"    k_caches = [{', '.join(f'k_cache_{i}' for i in range(self.n_full))}]")
            body.append(f"    v_caches = [{', '.join(f'v_cache_{i}' for i in range(self.n_full))}]")
        else:
            body.append("    k_caches = []")
            body.append("    v_caches = []")
            
        if self.n_lin > 0:
            body.append(f"    conv_states = [{', '.join(f'conv_state_{i}' for i in range(self.n_lin))}]")
            body.append(f"    rec_states = [{', '.join(f'rec_state_{i}' for i in range(self.n_lin))}]")
        else:
            body.append("    conv_states = []")
            body.append("    rec_states = []")
            
        body.append("    logits, k_out, v_out, conv_out, rec_out = self.model(input_ids, position_ids, k_caches, v_caches, conv_states, rec_states)")
        body.append("    return (logits, *k_out, *v_out, *conv_out, *rec_out)")
        
        code = f"def forward({args_str}):\n" + "\n".join(body)
        
        locs = {}
        exec(code, globals(), locs)
        import types
        self.forward = types.MethodType(locs["forward"], self)



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

    # Define input and output names in order
    input_names = ["input_ids", "position_ids"]
    for i in range(wrapper.n_full):
        input_names.append(f"k_cache_{i}")
    for i in range(wrapper.n_full):
        input_names.append(f"v_cache_{i}")
    for i in range(wrapper.n_lin):
        input_names.append(f"conv_state_{i}")
    for i in range(wrapper.n_lin):
        input_names.append(f"rec_state_{i}")

    output_names = ["logits"]
    for i in range(wrapper.n_full):
        output_names.append(f"k_cache_out_{i}")
    for i in range(wrapper.n_full):
        output_names.append(f"v_cache_out_{i}")
    for i in range(wrapper.n_lin):
        output_names.append(f"conv_state_out_{i}")
    for i in range(wrapper.n_lin):
        output_names.append(f"rec_state_out_{i}")

    # 5. Export to ONNX first, then convert to OpenVINO IR (prevents Linux PyTorch JIT frontend segfaults)
    onnx_path = output_dir / "model.onnx"
    print(f"\nExporting PyTorch model to intermediate ONNX at {onnx_path} ...")
    t0 = time.time()

    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "position_ids": {0: "batch_size", 1: "seq_len"},
        "logits": {0: "batch_size", 1: "seq_len"},
    }
    for i in range(wrapper.n_full):
        dynamic_axes[f"k_cache_{i}"] = {0: "batch_size", 2: "seq_len"}
        dynamic_axes[f"v_cache_{i}"] = {0: "batch_size", 2: "seq_len"}
        dynamic_axes[f"k_cache_out_{i}"] = {0: "batch_size", 2: "seq_len"}
        dynamic_axes[f"v_cache_out_{i}"] = {0: "batch_size", 2: "seq_len"}
    for i in range(wrapper.n_lin):
        dynamic_axes[f"conv_state_{i}"] = {0: "batch_size"}
        dynamic_axes[f"rec_state_{i}"] = {0: "batch_size"}
        dynamic_axes[f"conv_state_out_{i}"] = {0: "batch_size"}
        dynamic_axes[f"rec_state_out_{i}"] = {0: "batch_size"}

    torch.onnx.export(
        wrapper,
        example_inputs,
        str(onnx_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=18,
        do_constant_folding=True,
    )
    print(f"  ONNX export OK ({time.time()-t0:.1f}s)")

    print("\nConverting ONNX model to OpenVINO ...")
    t0 = time.time()
    ov_model = ov.convert_model(str(onnx_path))
    print(f"  convert_model OK ({time.time()-t0:.1f}s)")

    # Clean up the intermediate ONNX model to save disk space
    try:
        onnx_path.unlink(missing_ok=True)
    except Exception as e:
        print(f"Warning: could not delete temporary ONNX file {onnx_path}: {e}")

    # Set input and output names explicitly
    for idx, name in enumerate(input_names):
        ov_model.inputs[idx].set_names({name})
    for idx, name in enumerate(output_names):
        ov_model.outputs[idx].set_names({name})

    # Reshape to dynamic shapes
    print("  Applying dynamic shapes ...")
    dynamic_shapes = {}
    dynamic_shapes["input_ids"] = ov.PartialShape([-1, -1])
    dynamic_shapes["position_ids"] = ov.PartialShape([-1, -1])
    for i in range(wrapper.n_full):
        dynamic_shapes[f"k_cache_{i}"] = ov.PartialShape([-1, cfg.num_key_value_heads, -1, cfg.head_dim])
        dynamic_shapes[f"v_cache_{i}"] = ov.PartialShape([-1, cfg.num_key_value_heads, -1, cfg.head_dim])
    for i in range(wrapper.n_lin):
        dynamic_shapes[f"conv_state_{i}"] = ov.PartialShape([-1, cfg.linear_conv_dim, cfg.linear_conv_kernel_dim])
        dynamic_shapes[f"rec_state_{i}"] = ov.PartialShape([-1, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim])

    ov_model.reshape(dynamic_shapes)
    print(f"  OK ({time.time()-t0:.1f}s)")



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
        
        # Build inputs dict by matching string names to prevent crashes due to port reordering
        ov_inputs = {}
        print("\n--- Compile-check Debug Info ---")
        print("Expected input names order:", input_names)
        print("Compiled model inputs:")
        for idx, inp in enumerate(compiled.inputs):
            print(f"  Compiled Input {idx}: names={inp.get_names()}, any_name={inp.get_any_name()}, shape={inp.get_partial_shape()}, type={inp.get_element_type()}")
        
        for inp in compiled.inputs:
            name = inp.get_any_name()
            if name in input_names:
                idx = input_names.index(name)
                val = example_inputs[idx].numpy()
                ov_inputs[inp] = val
                print(f"  Mapped {name} -> array shape={val.shape}, dtype={val.dtype}")
            else:
                raise ValueError(f"Compiled model has unexpected input: {name}")

        print("Compiled model outputs:")
        for idx, out in enumerate(compiled.outputs):
            print(f"  Compiled Output {idx}: names={out.get_names()}, shape={out.get_partial_shape()}, type={out.get_element_type()}")

        print("Invoking compiled model...")
        sys.stdout.flush()
        ov_result = compiled(ov_inputs)
        print("Invocation successful!")

        # Retrieve logits safely by name
        logits_output = None
        for out in compiled.outputs:
            if "logits" in out.get_names():
                logits_output = out
                break
        if logits_output is None:
            logits_output = compiled.outputs[0]

        ov_logits = torch.from_numpy(ov_result[logits_output])
        diff      = (ref_logits.float() - ov_logits.float()).abs()
        pt_next   = ref_logits[0, -1].float().argmax().item()
        ov_next   = ov_logits[0, -1].float().argmax().item()
        print(f"  abs diff  max={diff.max().item():.3e}  mean={diff.mean().item():.3e}")
        print(f"  greedy next-token  PT={pt_next}  OV={ov_next}  match={pt_next == ov_next}")

    print("\nQwen3.5-4B OpenVINO conversion complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
