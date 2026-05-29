"""Convert toy DeepSeek-V4 (with KV-cache I/O) to OpenVINO IR, then run prefill + decode.

The model exposes past_key_values as L additional inputs (one per layer, shape
[B, S_past, hc_mult, dim]) and present_key_values as L additional outputs
(shape [B, S_past + S_new, hc_mult, dim]). S_past and S_new are both dynamic; S_past=0
gives prefill behavior bit-for-bit identical to scripts/convert_to_openvino.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import openvino as ov

from src import DeepseekV4ForCausalLM
from test_modeling_smoke import make_toy_config


def main():
    torch.manual_seed(0)
    cfg = make_toy_config()
    model = DeepseekV4ForCausalLM(cfg).eval()
    L = cfg.num_hidden_layers
    H = cfg.hc_mult
    dim = cfg.hidden_size

    B, S_prefill, n_decode = 1, 128, 4
    example_input_ids = torch.randint(0, cfg.vocab_size, (B, S_prefill))
    # Trace with zero-length past so the IR carries the full concat path but the example
    # data exercises prefill semantics. S_past becomes dynamic in the IR.
    empty_past = [torch.zeros(B, 0, H, dim, dtype=torch.float32) for _ in range(L)]

    print("=== PyTorch reference forward (prefill, use_cache=True) ===")
    with torch.inference_mode():
        ref_out = model(input_ids=example_input_ids, past_key_values=empty_past, use_cache=True)
    ref_logits = ref_out.logits
    ref_past = ref_out.past_key_values
    print(f"  ref logits shape: {tuple(ref_logits.shape)}, finite={torch.isfinite(ref_logits).all().item()}")
    print(f"  ref past[0] shape: {tuple(ref_past[0].shape)}")

    print("\n=== Tracing & converting to OpenVINO IR ===")
    class CacheWrapper(torch.nn.Module):
        """Flattens past_key_values from a list into L positional args; emits logits + L presents."""
        def __init__(self, m, n_layers):
            super().__init__()
            self.m = m
            self.n_layers = n_layers

        def forward(self, input_ids, *past):
            out = self.m(
                input_ids=input_ids,
                past_key_values=list(past),
                use_cache=True,
            )
            return (out.logits,) + tuple(out.past_key_values)

    wrapped = CacheWrapper(model, L).eval()

    past_names = [f"past_x_layer_{i}" for i in range(L)]
    present_names = [f"present_x_layer_{i}" for i in range(L)]

    # Provide shapes/types positionally (no names — torchscript trace numbers args).
    input_spec = [([-1, -1], ov.Type.i64)]
    input_spec += [([-1, -1, H, dim], ov.Type.f32) for _ in range(L)]

    ov_model = ov.convert_model(
        wrapped,
        example_input=(example_input_ids, *empty_past),
        input=input_spec,
    )

    # Assign stable names to input/output ports for runtime addressing.
    ov_model.inputs[0].set_names({"input_ids"})
    for i, name in enumerate(past_names):
        ov_model.inputs[i + 1].set_names({name})
    ov_model.outputs[0].set_names({"logits"})
    for i, name in enumerate(present_names):
        ov_model.outputs[i + 1].set_names({name})

    def _name_or(port):
        names = port.get_names()
        return next(iter(names)) if names else "<unnamed>"
    print(f"  IR inputs ({len(ov_model.inputs)}):")
    for p in ov_model.inputs:
        print(f"    {_name_or(p):>24}  {p.partial_shape}  {p.element_type}")
    print(f"  IR outputs ({len(ov_model.outputs)}):")
    for p in ov_model.outputs:
        print(f"    {_name_or(p):>24}  {p.partial_shape}  {p.element_type}")

    out_dir = ROOT / "ov_ir_toy"
    out_dir.mkdir(exist_ok=True)
    ir_path = out_dir / "deepseek_v4_toy_kv.xml"
    ov.save_model(ov_model, str(ir_path), compress_to_fp16=False)
    print(f"  saved IR to {ir_path}")

    print("\n=== Loading IR + running prefill on CPU ===")
    core = ov.Core()
    print(f"  available devices: {core.available_devices}")
    compiled = core.compile_model(ov_model, "CPU")

    feed = {"input_ids": example_input_ids.numpy().astype(np.int64)}
    for name, t in zip(past_names, empty_past):
        feed[name] = t.numpy()
    result = compiled(feed)
    ov_logits = torch.from_numpy(result["logits"])
    ov_past = [torch.from_numpy(result[n]) for n in present_names]

    print(f"  ov logits shape: {tuple(ov_logits.shape)} finite: {torch.isfinite(ov_logits).all().item()}")
    print(f"  ov past[0] shape: {tuple(ov_past[0].shape)}")

    diff = (ref_logits.float() - ov_logits.float()).abs()
    print(f"  prefill: abs diff max={diff.max().item():.4e} mean={diff.mean().item():.4e}")
    past_diffs = [(p_ref - p_ov).abs().max().item() for p_ref, p_ov in zip(ref_past, ov_past)]
    print(f"  past tensors: per-layer abs-max diff = {[f'{d:.2e}' for d in past_diffs]}")

    pt_next = ref_logits[0, -1].argmax().item()
    ov_next = ov_logits[0, -1].argmax().item()
    print(f"  greedy next-token: PT={pt_next}  OV={ov_next}  match={pt_next == ov_next}")
    assert pt_next == ov_next, "prefill greedy mismatch"

    print("\n=== Step-by-step decode over OpenVINO (cached) ===")
    past = ov_past
    decode_input_ids = torch.randint(0, cfg.vocab_size, (B, n_decode))
    ref_decode_logits_list = []
    ov_decode_logits_list = []
    for i in range(n_decode):
        # Reference (PyTorch cached decode for comparison)
        with torch.inference_mode():
            ref_step = model(
                input_ids=decode_input_ids[:, i:i + 1],
                past_key_values=ref_past,
                use_cache=True,
            )
        ref_decode_logits_list.append(ref_step.logits)
        ref_past = ref_step.past_key_values

        # OpenVINO cached decode
        feed = {"input_ids": decode_input_ids[:, i:i + 1].numpy().astype(np.int64)}
        for name, t in zip(past_names, past):
            feed[name] = t.numpy()
        result = compiled(feed)
        step_logits = torch.from_numpy(result["logits"])
        past = [torch.from_numpy(result[n]) for n in present_names]
        ov_decode_logits_list.append(step_logits)
        d = (ref_step.logits - step_logits).abs()
        ref_t = ref_step.logits[0, -1].argmax().item()
        ov_t = step_logits[0, -1].argmax().item()
        print(f"  step {i+1}: ov_shape={tuple(step_logits.shape)} past[0]_len={past[0].size(1)} "
              f"abs_max_vs_pt={d.max().item():.3e} ref_top={ref_t} ov_top={ov_t} match={ref_t == ov_t}")

    print("\nKV-CACHE CONVERT + RUN: PASSED")


if __name__ == "__main__":
    main()
