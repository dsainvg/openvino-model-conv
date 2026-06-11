# Qwen3.5-4B to OpenVINO

Dedicated conversion entry point for `Qwen/Qwen3.5-4B`.

This model is separate from the hand-ported `qwen36/` experiment. Its upstream
`config.json` currently reports:

```text
model_type = qwen3_5
architectures = ["Qwen3_5ForConditionalGeneration"]
```

Use the converter from the repo root:

```powershell
.\venv\Scripts\python.exe qwen35\scripts\convert_to_openvino.py --dtype fp16
```

For an already downloaded checkpoint:

```powershell
.\venv\Scripts\python.exe qwen35\scripts\convert_to_openvino.py --model C:\models\Qwen3.5-4B --local-files-only --dtype fp16
```

The script checks the installed `transformers` and `optimum-intel` exporter
support before downloading model weights. On the current repo venv
(`transformers==4.57.6`, `optimum-intel==1.27.0`), Optimum does not yet expose
an OpenVINO export config for `model_type=qwen3_5`, so the script stops before
the heavy download. With library versions that support `qwen3_5`, the same
script runs the OpenVINO export and optionally compiles the emitted IR files.
