"""Save toy DeepSeek-V4 in HF format + bundled OpenVINO IR, then load through optimum-intel.

Two-pronged integration test:
1. Save the toy PyTorch model + config as a Hugging Face model directory, with the
   configuration/modeling source files copied alongside and `auto_map` set so that
   `transformers.AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
   can re-instantiate it from disk.
2. Drop the already-converted OpenVINO IR (`openvino_model.xml`/.bin) into the same
   directory and load it via `optimum.intel.OVModelForCausalLM.from_pretrained`.

End result: any consumer with optimum-intel installed can do
    model = OVModelForCausalLM.from_pretrained("hf_export_toy")
and run inference on the toy model.
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src import DeepseekV4Config, DeepseekV4ForCausalLM
from test_modeling_smoke import make_toy_config


def main():
    torch.manual_seed(0)
    cfg = make_toy_config()

    # auto_map points at the bundled .py files so trust_remote_code can find them.
    cfg.auto_map = {
        "AutoConfig": "configuration_deepseek_v4.DeepseekV4Config",
        "AutoModelForCausalLM": "modeling_deepseek_v4.DeepseekV4ForCausalLM",
    }
    cfg.architectures = ["DeepseekV4ForCausalLM"]

    model = DeepseekV4ForCausalLM(cfg).eval()

    out_dir = ROOT / "hf_export_toy"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    print(f"=== Saving HF-format model to {out_dir} ===")
    model.save_pretrained(out_dir, safe_serialization=True)
    src_dir = ROOT / "src" / "deepseek_v4"
    for fname in ("configuration_deepseek_v4.py", "modeling_deepseek_v4.py"):
        shutil.copy(src_dir / fname, out_dir / fname)
    print(f"  files: {sorted(p.name for p in out_dir.iterdir())}")

    print("\n=== Reloading via transformers AutoModelForCausalLM (trust_remote_code) ===")
    from transformers import AutoConfig, AutoModelForCausalLM
    reloaded_cfg = AutoConfig.from_pretrained(out_dir, trust_remote_code=True)
    reloaded = AutoModelForCausalLM.from_pretrained(out_dir, trust_remote_code=True).eval()
    print(f"  reloaded type: {type(reloaded).__name__}")
    print(f"  param count: {sum(p.numel() for p in reloaded.parameters()):,}")

    B, S = 1, 64
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    with torch.inference_mode():
        ref = model(input_ids=input_ids).logits
        rel = reloaded(input_ids=input_ids).logits
    assert torch.allclose(ref, rel, atol=1e-5), "save/reload mismatch"
    print(f"  PT save/reload logits match: True")

    print("\n=== Bundling pre-converted OpenVINO IR ===")
    ir_src_xml = ROOT / "ov_ir_toy" / "deepseek_v4_toy.xml"
    ir_src_bin = ROOT / "ov_ir_toy" / "deepseek_v4_toy.bin"
    shutil.copy(ir_src_xml, out_dir / "openvino_model.xml")
    shutil.copy(ir_src_bin, out_dir / "openvino_model.bin")
    print(f"  copied IR -> {out_dir / 'openvino_model.xml'}")

    print("\n=== Loading via optimum.intel.OVModelForCausalLM.from_pretrained ===")
    from optimum.intel import OVModelForCausalLM
    # use_cache=False because the toy IR is prefill-only — no past_key_values input.
    ov_model = OVModelForCausalLM.from_pretrained(
        out_dir,
        trust_remote_code=True,
        compile=True,
        use_cache=False,
    )
    print(f"  ov_model type: {type(ov_model).__name__}")
    print(f"  device: {ov_model._device}")

    out = ov_model(input_ids=input_ids)
    ov_logits = out.logits if hasattr(out, "logits") else out[0]
    if not isinstance(ov_logits, torch.Tensor):
        ov_logits = torch.from_numpy(np.asarray(ov_logits))
    print(f"  ov logits shape: {tuple(ov_logits.shape)}")
    diff = (ref.float() - ov_logits.float()).abs()
    pt_top = int(ref[0, -1].argmax())
    ov_top = int(ov_logits[0, -1].argmax())
    print(f"  abs diff  max={diff.max().item():.4e}  mean={diff.mean().item():.4e}")
    print(f"  greedy:  PT={pt_top}  OV={ov_top}  match={pt_top == ov_top}")

    print("\nOPTIMUM-INTEL INTEGRATION: PASSED")


if __name__ == "__main__":
    main()
