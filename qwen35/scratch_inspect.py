import sys
import os
from pathlib import Path
import torch

# Force UTF-8 stdout encoding for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import openvino as ov
from src.configuration import make_toy_config
from src.modeling import QwenForCausalLM
from scripts.convert_to_openvino import FlatWrapper, build_example_inputs

def inspect():
    cfg = make_toy_config(
        num_layers=4,
        full_attention_interval=4,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        vocab_size=128,
    )
    model = QwenForCausalLM(cfg)
    model.eval()
    wrapper = FlatWrapper(model).eval()
    example_inputs = build_example_inputs(model, max_seq=8)
    
    # Export to ONNX
    onnx_path = REPO_ROOT / "temp_toy.onnx"
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

    print("Exporting ONNX...")
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
    
    print("Loading ONNX into OpenVINO...")
    ov_model = ov.convert_model(str(onnx_path))
    
    print("Inputs of ov_model:")
    for idx, inp in enumerate(ov_model.inputs):
        print(f"  Input {idx}: names={inp.get_names()}, shape={inp.get_partial_shape()}")
        
    print("Outputs of ov_model:")
    for idx, out in enumerate(ov_model.outputs):
        print(f"  Output {idx}: names={out.get_names()}, shape={out.get_partial_shape()}")

    print("Compiling model...")
    core = ov.Core()
    compiled = core.compile_model(ov_model, "CPU")
    
    print("Inputs of compiled model:")
    for idx, inp in enumerate(compiled.inputs):
        print(f"  Input {idx}: names={inp.get_names()}, shape={inp.get_partial_shape()}")
        
    print("Outputs of compiled model:")
    for idx, out in enumerate(compiled.outputs):
        print(f"  Output {idx}: names={out.get_names()}, shape={out.get_partial_shape()}")

    onnx_path.unlink(missing_ok=True)

if __name__ == "__main__":
    inspect()
