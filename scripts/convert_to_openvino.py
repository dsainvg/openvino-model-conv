"""Convert toy DeepSeek-V4 to OpenVINO IR via openvino.convert_model, then load + run.

This is the PoC end-to-end test: prove the V4 PyTorch model traces and runs through OpenVINO.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
import torch
import openvino as ov

from deepseek_v4 import DeepseekV4ForCausalLM
from test_modeling_smoke import make_toy_config


def main():
    torch.manual_seed(0)
    cfg = make_toy_config()
    model = DeepseekV4ForCausalLM(cfg).eval()

    B, S = 1, 128
    example_input_ids = torch.randint(0, cfg.vocab_size, (B, S))

    print("=== PyTorch reference forward ===")
    with torch.inference_mode():
        ref_logits = model(input_ids=example_input_ids).logits
    print(f"  ref logits shape: {tuple(ref_logits.shape)}, finite={torch.isfinite(ref_logits).all().item()}")

    print("\n=== Tracing & converting to OpenVINO IR ===")
    # Use a wrapper that returns just the logits tensor (convert_model handles tensor-out best).
    class LogitsOnly(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, input_ids): return self.m(input_ids=input_ids).logits

    wrapped = LogitsOnly(model).eval()

    ov_model = ov.convert_model(
        wrapped,
        example_input=(example_input_ids,),
        input=[("input_ids", [-1, -1], ov.Type.i64)],
    )
    def _name_or(port):
        names = port.get_names()
        return next(iter(names)) if names else "<unnamed>"
    print(f"  IR inputs:  {[(_name_or(p), p.partial_shape, p.element_type) for p in ov_model.inputs]}")
    print(f"  IR outputs: {[(_name_or(p), p.partial_shape, p.element_type) for p in ov_model.outputs]}")
    # Name the output so downstream tools/runtime have a stable handle.
    if not ov_model.outputs[0].get_names():
        ov_model.outputs[0].set_names({"logits"})

    out_dir = ROOT / "ov_ir_toy"
    out_dir.mkdir(exist_ok=True)
    ir_path = out_dir / "deepseek_v4_toy.xml"
    ov.save_model(ov_model, str(ir_path), compress_to_fp16=False)
    print(f"  saved IR to {ir_path}")

    print("\n=== Loading IR + running inference ===")
    core = ov.Core()
    print(f"  available devices: {core.available_devices}")
    compiled = core.compile_model(ov_model, "CPU")
    out_tensor = compiled([example_input_ids.numpy()])[0]
    ov_logits = torch.from_numpy(out_tensor)
    print(f"  ov logits shape: {tuple(ov_logits.shape)}")
    print(f"  ov logits finite: {torch.isfinite(ov_logits).all().item()}")

    print("\n=== Comparing PyTorch vs OpenVINO ===")
    diff = (ref_logits.float() - ov_logits.float()).abs()
    rel = diff / (ref_logits.float().abs() + 1e-6)
    print(f"  abs diff  max={diff.max().item():.6e}  mean={diff.mean().item():.6e}")
    print(f"  rel diff  max={rel.max().item():.6e}  mean={rel.mean().item():.6e}")
    # Sanity: greedy next-token prediction should match.
    pt_next = ref_logits[0, -1].argmax().item()
    ov_next = ov_logits[0, -1].argmax().item()
    print(f"  greedy next-token: PT={pt_next}  OV={ov_next}  match={pt_next == ov_next}")

    print("\nCONVERT + RUN: PASSED")


if __name__ == "__main__":
    main()
