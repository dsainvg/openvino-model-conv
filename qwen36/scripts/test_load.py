import time, traceback, sys, os
import torch

MODEL_DIR = r"C:\Users\intel\models\qwen36-35b-int4"

print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())

from transformers import AutoModelForCausalLM, AutoTokenizer

t0 = time.time()
print("loading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
print(f"  tokenizer ok ({time.time()-t0:.1f}s) vocab={len(tok)}")

t0 = time.time()
print("loading model (low_cpu_mem_usage=True, dtype=bfloat16, device_map=cpu)...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print(f"  model ok ({time.time()-t0:.1f}s)")
    print("  num params:", sum(p.numel() for p in model.parameters()) / 1e9, "B")
except Exception as e:
    print(f"  FAILED after {time.time()-t0:.1f}s")
    print("---")
    traceback.print_exc()
    sys.exit(1)
