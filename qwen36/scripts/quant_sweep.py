"""Phase 3.1 quantization sweep for a Qwen3.6 routed expert.

Takes ONE real routed expert (dequantized from the GPTQ checkpoint to FP32),
converts it to an OpenVINO IR, then runs NNCF weight compression across
several precisions and reports, for each:

  - IR size on disk (xml + bin) and compression ratio vs FP32
  - activation reconstruction error vs the FP32 IR on random inputs
    (max abs diff, relative RMSE, cosine similarity)

Activation error is a perplexity *proxy*: we can't run WikiText-2 PPL on this
Windows host (no gptqmodel, no BF16 reference), but how much a precision
perturbs an expert's output on representative inputs is a meaningful,
reproducible quality signal.

If the real checkpoint is absent, falls back to a random-weight expert so the
sweep mechanics still run (sizes are valid; absolute error magnitudes are not
representative).

Run from venv-qwen.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import nncf  # noqa: E402
import openvino as ov  # noqa: E402

from src.configuration import Qwen36Config  # noqa: E402
from src.modeling import QwenExpertFFN  # noqa: E402

MODEL_DIR = Path(r"C:\Users\intel\models\qwen36-35b-int4")

# (label, mode, group_size) -- group_size None means per-channel / default.
SWEEP = [
    ("INT8_ASYM", nncf.CompressWeightsMode.INT8_ASYM, None),
    ("INT4_ASYM", nncf.CompressWeightsMode.INT4_ASYM, 32),
    ("INT4_SYM", nncf.CompressWeightsMode.INT4_SYM, 32),
    ("NF4", nncf.CompressWeightsMode.NF4, 32),
    ("MXFP4", nncf.CompressWeightsMode.MXFP4, 32),
]


def _build_real_expert(cfg: Qwen36Config) -> tuple[QwenExpertFFN, bool]:
    if MODEL_DIR.exists() and (MODEL_DIR / "model.safetensors.index.json").exists():
        from src.gptq_dequant import build_shard_index, load_gptq_linear

        wm = build_shard_index(MODEL_DIR)
        expert = QwenExpertFFN(cfg.hidden_size, cfg.moe_intermediate_size)
        prefix = "model.language_model.layers.0.mlp.experts.0"
        with torch.no_grad():
            expert.gate_proj.weight.copy_(load_gptq_linear(wm, MODEL_DIR, f"{prefix}.gate_proj"))
            expert.up_proj.weight.copy_(load_gptq_linear(wm, MODEL_DIR, f"{prefix}.up_proj"))
            expert.down_proj.weight.copy_(load_gptq_linear(wm, MODEL_DIR, f"{prefix}.down_proj"))
        expert.eval()
        return expert, True
    torch.manual_seed(0)
    expert = QwenExpertFFN(cfg.hidden_size, cfg.moe_intermediate_size)
    expert.eval()
    return expert, False


def _ir_size(path: Path) -> int:
    return path.stat().st_size + path.with_suffix(".bin").stat().st_size


def _errors(ref: torch.Tensor, got: torch.Tensor) -> tuple[float, float, float]:
    diff = (ref - got).float()
    max_abs = diff.abs().max().item()
    rel_rmse = (diff.pow(2).mean().sqrt() / (ref.float().pow(2).mean().sqrt() + 1e-9)).item()
    cos = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), got.float().flatten(), dim=0
    ).item()
    return max_abs, rel_rmse, cos


def main() -> int:
    cfg = Qwen36Config.from_pretrained_dir(MODEL_DIR) if MODEL_DIR.exists() else Qwen36Config()
    expert, real = _build_real_expert(cfg)
    print("=" * 78)
    print(f"Qwen3.6 expert quantization sweep  ({'REAL layer0/expert0' if real else 'RANDOM weights'})")
    print(f"expert: hidden={cfg.hidden_size} intermediate={cfg.moe_intermediate_size} "
          f"params={sum(p.numel() for p in expert.parameters()):,}")
    print("=" * 78)

    core = ov.Core()
    torch.manual_seed(1)
    probe = torch.randn(16, cfg.hidden_size)  # 16 representative tokens
    with torch.no_grad():
        ref_torch = expert(probe)

    out_dir = REPO / "ov_ir_qwen36_toy" / "quant_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    fp32_xml = out_dir / "fp32.xml"
    ov_model = ov.convert_model(expert, example_input=probe)
    ov.save_model(ov_model, fp32_xml, compress_to_fp16=False)
    fp32_size = _ir_size(fp32_xml)

    # FP32 OV baseline output
    comp = core.compile_model(str(fp32_xml), "CPU")
    ref_ov = torch.from_numpy(comp({comp.inputs[0]: probe.numpy()})[comp.outputs[0]])
    base_max, base_rmse, base_cos = _errors(ref_torch, ref_ov)

    print(f"\nFP32 IR: {fp32_size/1024:8.1f} KB   (torch vs OV-fp32: "
          f"max_abs={base_max:.2e} cos={base_cos:.6f})")
    print(f"\n{'precision':<11}{'size KB':>10}{'ratio':>8}{'max_abs':>12}{'rel_rmse':>12}{'cosine':>11}")
    print("-" * 78)
    print(f"{'FP32':<11}{fp32_size/1024:>10.1f}{1.0:>8.2f}{0.0:>12.2e}{0.0:>12.2e}{1.0:>11.6f}")

    for label, mode, gs in SWEEP:
        model_in = core.read_model(str(fp32_xml))
        kwargs = {"mode": mode}
        if gs is not None:
            kwargs["group_size"] = gs
            kwargs["ratio"] = 1.0
        try:
            compressed = nncf.compress_weights(model_in, **kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"{label:<11}{'--':>10}  FAILED: {type(e).__name__}: {str(e)[:40]}")
            continue
        out_xml = out_dir / f"{label}.xml"
        ov.save_model(compressed, out_xml, compress_to_fp16=False)
        size = _ir_size(out_xml)

        comp_c = core.compile_model(str(out_xml), "CPU")
        got = torch.from_numpy(comp_c({comp_c.inputs[0]: probe.numpy()})[comp_c.outputs[0]])
        max_abs, rel_rmse, cos = _errors(ref_ov, got)
        print(f"{label:<11}{size/1024:>10.1f}{fp32_size/size:>8.2f}"
              f"{max_abs:>12.2e}{rel_rmse:>12.2e}{cos:>11.6f}")

    print("-" * 78)
    print("error columns are vs the FP32 OV output (activation-space perplexity proxy).")
    print("note: real experts are already GPTQ-Int4 on disk; this sweep shows what an")
    print("      on-device NNCF re-quant of the dequantized expert would cost in quality.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
