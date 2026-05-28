"""Convert the toy Qwen3.6 model to an OpenVINO IR and verify numerical match
against the PyTorch source.

Run from venv-qwen (has openvino 2026.1.0).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import openvino as ov  # noqa: E402

from src.qwen36.configuration_qwen36 import make_toy_config  # noqa: E402
from src.qwen36.modeling_qwen36 import QwenForCausalLM  # noqa: E402


class FlatWrapper(nn.Module):
    """Flattens the state-list args of QwenForCausalLM into positional tensors
    so each one becomes a distinct OV graph input/output."""

    def __init__(self, model: QwenForCausalLM):
        super().__init__()
        self.model = model
        self.n_full = model.model.num_full_layers
        self.n_lin = model.model.num_linear_layers

    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # input_ids, position_ids, then 2*n_full k/v + 2*n_lin conv/rec
        input_ids, position_ids = args[0], args[1]
        i = 2
        k_caches = list(args[i : i + self.n_full]); i += self.n_full
        v_caches = list(args[i : i + self.n_full]); i += self.n_full
        conv_states = list(args[i : i + self.n_lin]); i += self.n_lin
        rec_states = list(args[i : i + self.n_lin]); i += self.n_lin
        logits, k_out, v_out, conv_out, rec_out = self.model(
            input_ids, position_ids, k_caches, v_caches, conv_states, rec_states
        )
        return (logits, *k_out, *v_out, *conv_out, *rec_out)


def build_example_inputs(model: QwenForCausalLM, batch: int = 1, max_seq: int = 8):
    state = model.empty_state(batch=batch, max_seq=max_seq)
    return (
        torch.tensor([[1]] * batch, dtype=torch.long),  # input_ids
        torch.tensor([[0]] * batch, dtype=torch.long),  # position_ids
        *state["k_caches"],
        *state["v_caches"],
        *state["conv_states"],
        *state["rec_states"],
    )


def main() -> int:
    out_dir = Path(__file__).resolve().parents[1] / "ov_ir_qwen36_toy"
    out_dir.mkdir(exist_ok=True)
    ir_xml = out_dir / "toy.xml"

    cfg = make_toy_config()
    print("toy cfg:", cfg.layer_types, "experts:", cfg.num_experts, "hidden:", cfg.hidden_size)

    torch.manual_seed(0)
    model = QwenForCausalLM(cfg)
    model.eval()

    wrapper = FlatWrapper(model)
    example_inputs = build_example_inputs(model)

    # PyTorch reference outputs first
    with torch.no_grad():
        torch_outputs = wrapper(*example_inputs)
    torch_logits = torch_outputs[0]
    print(f"torch logits shape={tuple(torch_logits.shape)} mean={torch_logits.mean().item():+.4e} std={torch_logits.std().item():+.4e}")

    # Convert to OV
    print("\nov.convert_model ...")
    t0 = time.time()
    ov_model = ov.convert_model(wrapper, example_input=example_inputs)
    print(f"  convert_model OK ({time.time()-t0:.1f}s)")

    # Save IR
    ov.save_model(ov_model, ir_xml)
    print(f"  saved IR to {ir_xml}")

    # Re-load and compile for CPU
    core = ov.Core()
    compiled = core.compile_model(str(ir_xml), "CPU")
    print(f"  compiled on CPU, {len(compiled.inputs)} inputs / {len(compiled.outputs)} outputs")

    # Run OV on the same example inputs
    ov_inputs = {compiled.inputs[i]: example_inputs[i].numpy() for i in range(len(compiled.inputs))}
    result = compiled(ov_inputs)
    ov_logits = torch.from_numpy(result[compiled.outputs[0]])
    print(f"\nov logits shape={tuple(ov_logits.shape)} mean={ov_logits.mean().item():+.4e} std={ov_logits.std().item():+.4e}")

    # Numerical comparison
    diff = (ov_logits - torch_logits).abs()
    print(f"\ntorch vs OV abs-diff: max={diff.max().item():.3e} mean={diff.mean().item():.3e}")
    rel_diff = diff.max() / (torch_logits.abs().max() + 1e-9)
    print(f"relative max-diff: {rel_diff.item():.3e}")
    # Tolerance: OV applies graph fusions (e.g. layout reordering, MatMul +
    # bias fusion) that introduce small float reordering noise vs eager
    # PyTorch. For a 2-layer toy with std~0.5 outputs, ~1e-3 max-diff is the
    # typical floor in our setup.
    ok = bool(torch.allclose(torch_logits, ov_logits, atol=1e-3, rtol=1e-3))
    print(f"\nresult: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
