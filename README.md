SDXL Flask Web UI

A minimal Flask-based web UI for Stable Diffusion XL (SDXL) supporting:
- Text to Image (SDXL base)
- Image to Image (SDXL img2img)
- 4x Upscaling (SD x4 Upscaler)

Requirements

- Python 3.10+
- GPU recommended (CUDA) or Apple Silicon (MPS). CPU will work but is slow.
- Disk space ~15GB to download models on first run

Setup

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

Environment variables (optional)

- SDXL_MODEL_ID (default: stabilityai/stable-diffusion-xl-base-1.0)
- SD_UPSCALE_MODEL_ID (default: stabilityai/stable-diffusion-x4-upscaler)
- FLASK_HOST (default: 0.0.0.0)
- FLASK_PORT (default: 5000)
- FLASK_DEBUG (default: 0)

Run

```
python app.py
```

Open http://localhost:5000

Notes

- First run downloads large models; allow time and bandwidth.
- On CUDA, xFormers is enabled for memory-efficient attention.
- On Apple Silicon, MPS is used automatically.
- Outputs are saved to static/outputs/

Troubleshooting

- If PyTorch fails to use GPU, check CUDA/MPS availability in Python:

```
import torch
print('cuda', torch.cuda.is_available())
print('mps', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
```

- If you hit out-of-memory errors, try reducing width/height or steps.
- For CPU-only, expect slower generations and higher memory usage.

Offline usage

- Pre-download models into local folders and run fully offline:

```
python scripts/download_models.py
```

This will create a `models/` directory with:
- `models/sdxl` containing the SDXL base
- `models/x4-upscaler` containing the 4x upscaler

Then set environment variables before running:

```
export SDXL_LOCAL_DIR="$(pwd)/models/sdxl"
export SD_UPSCALE_LOCAL_DIR="$(pwd)/models/x4-upscaler"
export HF_HUB_OFFLINE=1
python app.py
```

When `HF_HUB_OFFLINE=1` is set, the app loads models only from local files and will not attempt network access.


