"""Download config + modeling code for DeepSeek-V4-Flash, no weights."""
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download

REPO = "deepseek-ai/DeepSeek-V4-Flash"
DEST = Path(r"C:\Users\intel\dpv4-openvino\v4_flash_meta")
DEST.mkdir(parents=True, exist_ok=True)

api = HfApi()
info = api.model_info(REPO)
print("All files in repo:")
for s in info.siblings:
    print(f"  {s.rfilename}")

# Pull anything that isn't a weight blob.
SKIP_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx")
print("\nDownloading non-weight files:")
for s in info.siblings:
    name = s.rfilename
    if name.endswith(SKIP_SUFFIXES):
        continue
    try:
        path = hf_hub_download(REPO, name, local_dir=str(DEST))
        print(f"  OK   {name}")
    except Exception as e:
        print(f"  FAIL {name}  {type(e).__name__}: {str(e)[:120]}")

print("\nDONE. Local meta dir:", DEST)
