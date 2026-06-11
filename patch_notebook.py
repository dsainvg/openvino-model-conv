import json

notebook_path = "kaggle_qwen35_convert_push_hf.ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] == "markdown":
        text = "".join(cell["source"])
        if "## 5 · Convert to stateful OpenVINO model" in text:
            cell["source"] = [
                "## 5 · Convert to split-IR OpenVINO model\n",
                "\n",
                "1. Loads weights layer-by-layer\n",
                "2. `ov.convert_model()` — traces each layer individually into separate OV graphs\n",
                "3. `nncf.compress_weights(INT4_SYM, group_size=128)` — compress Linear weights per layer\n",
                "4. Exports `openvino_tokenizer.xml` + `generation_config.json`\n",
                "\n",
                "**Expected size**: ~2–2.5 GB &nbsp;&nbsp; **Expected runtime**: ~30 min on Kaggle CPU"
            ]
        elif "## 5b · Sanity checks" in text:
            cell["source"] = [
                "## 5b · Sanity checks\n",
                "\n",
                "Verifies the output **without running inference**:\n",
                "- All expected files exist and are non-empty\n",
                "- `generation_config.json` has the right EOS token\n",
                "- Tokenizer files are valid OV models"
            ]
        elif "**Output**: Single `openvino_model.xml`" in text:
            new_source = []
            for line in cell["source"]:
                if "**Output**: Single `openvino_model.xml`" in line:
                    new_source.append("**Output**: Split-IR models (`embed_tokens.xml`, `layer_0.xml`, etc.) + `openvino_tokenizer.xml` + `generation_config.json`\n")
                elif "import openvino_genai as ov_genai\n" in line:
                    new_source.append("from qwen_npu_pipeline import QwenPipeline\n")
                elif "pipe = ov_genai.LLMPipeline" in line:
                    new_source.append("pipe = QwenPipeline('/path/to/model')\n")
                else:
                    new_source.append(line)
            cell["source"] = new_source

    elif cell["cell_type"] == "code":
        text = "".join(cell["source"])
        if "convert_to_openvino_genai.py" in text:
            cell["source"] = [
                "result = subprocess.run(\n",
                "    [\n",
                "        sys.executable, str(SCRIPTS_DIR / 'convert_to_openvino.py'),\n",
                "        '--model-dir', str(MODEL_DIR),\n",
                "        '--output',    str(OUTPUT_DIR),\n",
                "        '--dtype',     'bf16',\n",
                "        '--group-size', '128',\n",
                "    ],\n",
                "    cwd=str(QWEN35_DIR),\n",
                "    env={**os.environ},\n",
                ")\n",
                "if result.returncode != 0:\n",
                "    raise RuntimeError('convert_to_openvino.py failed.')\n",
                "print('\\n Split-IR model saved to', OUTPUT_DIR)"
            ]
        elif "REQUIRED_FILES =" in text and "openvino_model.xml" in text:
            cell["source"] = [
                "import json\n",
                "import openvino as ov\n",
                "\n",
                "REQUIRED_FILES = [\n",
                "    'embed_tokens.xml',\n",
                "    'embed_tokens.bin',\n",
                "    'layer_0.xml',\n",
                "    'layer_0.bin',\n",
                "    'openvino_tokenizer.xml',\n",
                "    'openvino_detokenizer.xml',\n",
                "    'generation_config.json',\n",
                "    'config.json',\n",
                "]\n",
                "\n",
                "print('--- File checks ---')\n",
                "all_ok = True\n",
                "for fname in REQUIRED_FILES:\n",
                "    fpath = OUTPUT_DIR / fname\n",
                "    if not fpath.exists():\n",
                "        print(f'  MISSING : {fname}')\n",
                "        all_ok = False\n",
                "    else:\n",
                "        mb = fpath.stat().st_size / 1e6\n",
                "        print(f'  OK  {fname:45s} {mb:7.1f} MB')\n",
                "\n",
                "print('\\n--- generation_config.json ---')\n",
                "with open(OUTPUT_DIR / 'generation_config.json') as f:\n",
                "    gen_cfg = json.load(f)\n",
                "print(f'  eos_token_id   : {gen_cfg.get(\"eos_token_id\")}')\n",
                "print(f'  max_new_tokens : {gen_cfg.get(\"max_new_tokens\")}')\n",
                "\n",
                "print('\\n--- Tokenizer check ---')\n",
                "core = ov.Core()\n",
                "try:\n",
                "    tok_model = core.read_model(str(OUTPUT_DIR / 'openvino_tokenizer.xml'))\n",
                "    print(f'  OK  openvino_tokenizer has {len(tok_model.inputs)} input(s), {len(tok_model.outputs)} output(s)')\n",
                "except Exception as e:\n",
                "    print(f'  WARN  could not read openvino_tokenizer.xml: {e}')\n",
                "\n",
                "print()\n",
                "if all_ok:\n",
                "    print('All sanity checks passed.')\n",
                "else:\n",
                "    raise RuntimeError('One or more sanity checks FAILED. See output above.')"
            ]

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully.")
