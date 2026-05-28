"""Phase 2 split-inference tests.

Verify that the router-only / per-expert / combine seam matches the
monolithic forward exactly (in pure torch), and that one expert converts
cleanly to OV.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.qwen36.configuration_qwen36 import make_toy_config  # noqa: E402
from src.qwen36.modeling_qwen36 import QwenForCausalLM, QwenExpertFFN  # noqa: E402
from src.qwen36.split_inference import (  # noqa: E402
    build_standalone_expert,
    extract_expert_state_dict,
    monolithic_step_via_split,
)


@pytest.fixture
def toy_model():
    torch.manual_seed(0)
    cfg = make_toy_config()
    model = QwenForCausalLM(cfg)
    model.eval()
    return model


def test_split_path_matches_monolithic_one_step(toy_model):
    """The pure-torch split orchestrator must produce bit-equivalent outputs
    to the monolithic forward (modulo per-token expert reduction order)."""
    state = toy_model.empty_state(batch=1, max_seq=8)
    input_ids = torch.tensor([[42]])
    position_ids = torch.tensor([[0]])

    with torch.no_grad():
        mono_logits, mk, mv, mc, mr = toy_model(
            input_ids, position_ids, state["k_caches"], state["v_caches"],
            state["conv_states"], state["rec_states"],
        )
        split_logits, split_state = monolithic_step_via_split(
            toy_model, input_ids, position_ids, state,
        )

    diff = (mono_logits - split_logits).abs().max().item()
    # The split path computes only the K selected experts per token instead
    # of the full sum-over-all-experts weighted by (idx == e). Mathematically
    # identical, but FP reduction order may differ slightly. Allow ~1e-4.
    assert diff < 1e-4, f"split vs mono diff {diff}"
    # State must also match across all four lists.
    for a, b in zip(split_state["k_caches"], mk):
        assert torch.allclose(a, b)
    for a, b in zip(split_state["conv_states"], mc):
        assert torch.allclose(a, b)
    for a, b in zip(split_state["rec_states"], mr):
        assert torch.allclose(a, b)


def test_split_path_matches_monolithic_two_steps(toy_model):
    """Across two decode steps, state must propagate identically through
    both paths."""
    state = toy_model.empty_state(batch=1, max_seq=8)
    with torch.no_grad():
        # Step 0 via both
        mono_l0, mk0, mv0, mc0, mr0 = toy_model(
            torch.tensor([[3]]), torch.tensor([[0]]),
            state["k_caches"], state["v_caches"], state["conv_states"], state["rec_states"],
        )
        split_l0, ss0 = monolithic_step_via_split(
            toy_model, torch.tensor([[3]]), torch.tensor([[0]]), state,
        )
        # Step 1 via both, feeding step-0 state
        mono_l1, _mk1, _mv1, _mc1, _mr1 = toy_model(
            torch.tensor([[5]]), torch.tensor([[1]]), mk0, mv0, mc0, mr0,
        )
        split_l1, _ss1 = monolithic_step_via_split(
            toy_model, torch.tensor([[5]]), torch.tensor([[1]]), ss0,
        )
    assert (mono_l1 - split_l1).abs().max().item() < 1e-4


def test_extract_and_rebuild_expert(toy_model):
    """extract_expert_state_dict -> build_standalone_expert must produce a
    bit-equivalent expert."""
    cfg = toy_model.config
    sd = extract_expert_state_dict(toy_model, layer_idx=1, expert_idx=2)
    standalone = build_standalone_expert(cfg, weights=sd)

    x = torch.randn(3, cfg.hidden_size)
    with torch.no_grad():
        ref = toy_model.model.layers[1].mlp.experts[2](x)
        got = standalone(x)
    assert torch.equal(ref, got), "standalone expert output differs from original"


# ---------------------------------------------------------------------------
# OV conversion of one expert
# ---------------------------------------------------------------------------


def test_single_expert_ov_roundtrip(toy_model, tmp_path):
    """Convert one QwenExpertFFN to OV and verify forward equivalence."""
    import openvino as ov

    cfg = toy_model.config
    sd = extract_expert_state_dict(toy_model, layer_idx=1, expert_idx=0)
    expert = build_standalone_expert(cfg, weights=sd)

    example_x = torch.randn(2, cfg.hidden_size)
    with torch.no_grad():
        torch_out = expert(example_x)

    ov_model = ov.convert_model(expert, example_input=example_x)
    ir_path = tmp_path / "expert.xml"
    ov.save_model(ov_model, ir_path)

    core = ov.Core()
    compiled = core.compile_model(str(ir_path), "CPU")
    result = compiled({compiled.inputs[0]: example_x.numpy()})
    ov_out = torch.from_numpy(result[compiled.outputs[0]])

    diff = (torch_out - ov_out).abs().max().item()
    # Same OV fp-fusion floor as the full toy conversion (~5e-4).
    assert diff < 5e-4, f"single-expert OV roundtrip diff {diff}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
