"""Probe Hugging Face for DeepSeek V4 repos and pull config-only metadata."""
from huggingface_hub import HfApi, hf_hub_download

CANDIDATES = [
    "deepseek-ai/DeepSeek-V4",
    "deepseek-ai/DeepSeek-V4-Pro",
    "deepseek-ai/DeepSeek-V4-Flash",
    "deepseek-ai/DeepSeek-V4-Preview",
    "deepseek-ai/DeepSeek-V4-Pro-Preview",
    "deepseek-ai/DeepSeek-V4-Flash-Preview",
]

api = HfApi()
found = []
for repo in CANDIDATES:
    try:
        info = api.model_info(repo)
        found.append(repo)
        print(f"FOUND  {repo}  files={len(info.siblings)}  pipeline={info.pipeline_tag}")
    except Exception as e:
        print(f"MISS   {repo}  {type(e).__name__}: {str(e)[:120]}")

print("---")
print("FOUND_REPOS:", found)
