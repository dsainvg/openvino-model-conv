"""KV-cache equivalence test: compare full prefill against step-by-step decode with cache.

For each step in the toy generation, the decode-with-cache output must match the
prefill-on-prefix output (modulo small float reordering). This is the correctness bar
for the cached path.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import torch

from deepseek_v4 import DeepseekV4ForCausalLM
from test_modeling_smoke import make_toy_config


def main():
    torch.manual_seed(0)
    cfg = make_toy_config()
    model = DeepseekV4ForCausalLM(cfg).eval()

    # Use S=128 so the ratio=128 compressor produces at least one chunk.
    B, S_prefill = 1, 128
    n_decode = 4
    input_ids = torch.randint(0, cfg.vocab_size, (B, S_prefill + n_decode))

    print("=== Reference: single forward over the full sequence ===")
    with torch.inference_mode():
        ref = model(input_ids=input_ids, use_cache=False)
    ref_logits = ref.logits
    print(f"  ref logits shape: {tuple(ref_logits.shape)}")

    print("\n=== Cached path: prefill then step-by-step decode ===")
    with torch.inference_mode():
        out = model(input_ids=input_ids[:, :S_prefill], use_cache=True)
    prefill_logits = out.logits
    past = out.past_key_values
    print(f"  prefill logits shape: {tuple(prefill_logits.shape)}")
    print(f"  past_key_values: {len(past)} layers, layer-0 shape: {tuple(past[0].shape)}")

    # Check prefill segment matches reference.
    diff_prefill = (ref_logits[:, :S_prefill] - prefill_logits).abs()
    print(f"  prefill vs ref-prefill: abs max={diff_prefill.max().item():.3e} mean={diff_prefill.mean().item():.3e}")

    # Now decode one token at a time.
    decode_logits_list = []
    for i in range(n_decode):
        pos = S_prefill + i
        with torch.inference_mode():
            out = model(
                input_ids=input_ids[:, pos:pos + 1],
                past_key_values=past,
                use_cache=True,
            )
        decode_logits_list.append(out.logits)  # [B, 1, V]
        past = out.past_key_values
        diff = (ref_logits[:, pos:pos + 1] - out.logits).abs()
        print(f"  step {i+1}: decode logits {tuple(out.logits.shape)}, "
              f"past layer-0 shape {tuple(past[0].shape)}, "
              f"vs ref abs max={diff.max().item():.3e} mean={diff.mean().item():.3e}")

    decode_logits = torch.cat(decode_logits_list, dim=1)  # [B, n_decode, V]
    diff_total = (ref_logits[:, S_prefill:] - decode_logits).abs()
    print(f"\n  decode-cache vs ref (over {n_decode} decoded steps):")
    print(f"    abs diff  max={diff_total.max().item():.3e}  mean={diff_total.mean().item():.3e}")

    ref_top = ref_logits[:, S_prefill:].argmax(dim=-1)
    dec_top = decode_logits.argmax(dim=-1)
    match = bool((ref_top == dec_top).all())
    print(f"    greedy next-token match across all {n_decode} steps: {match}")

    # Tolerance: with fp32 throughout we expect <1e-3 abs diff; fail loudly otherwise.
    assert match, "greedy next-token mismatch between cached decode and reference"
    assert diff_total.max().item() < 1e-3, f"abs diff exceeds 1e-3: {diff_total.max().item()}"
    print("\nKV-CACHE EQUIVALENCE: PASSED")


if __name__ == "__main__":
    main()
