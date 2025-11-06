# Path to aria2c.exe
$aria2 = "D:\aria2-1.37.0-win-64bit-build1\aria2\aria2c.exe"

# Base output directory
$outdir = "D:\University\Personal Project\Capstoneflask\models\sdxl"

# Create subfolders if not exist
New-Item -ItemType Directory -Force -Path "$outdir\unet" | Out-Null
New-Item -ItemType Directory -Force -Path "$outdir\vae" | Out-Null
New-Item -ItemType Directory -Force -Path "$outdir\text_encoder" | Out-Null
New-Item -ItemType Directory -Force -Path "$outdir\scheduler" | Out-Null
New-Item -ItemType Directory -Force -Path "$outdir\tokenizer" | Out-Null

# ---------------- Weights ----------------

# 1. UNet
& $aria2 -x 16 -s 16 -k 1M -c `
  -d "$outdir\unet" `
  -o diffusion_pytorch_model.safetensors `
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/diffusion_pytorch_model.safetensors"

# 2. VAE
& $aria2 -x 16 -s 16 -k 1M -c `
  -d "$outdir\vae" `
  -o diffusion_pytorch_model.safetensors `
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/vae/diffusion_pytorch_model.safetensors"

# 3. Text Encoder 1
& $aria2 -x 16 -s 16 -k 1M -c `
  -d "$outdir\text_encoder" `
  -o model.safetensors `
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/model.safetensors"

# ---------------- Configs ----------------

# model_index.json
& $aria2 -c -d $outdir -o model_index.json `
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/model_index.json"

# Scheduler config
& $aria2 -c -d "$outdir\scheduler" -o scheduler_config.json `
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/scheduler/scheduler_config.json"

# ---------------- Tokenizer ----------------

& $aria2 -c -d "$outdir\tokenizer" -o tokenizer_config.json `
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer/tokenizer_config.json"

& $aria2 -c -d "$outdir\tokenizer" -o special_tokens_map.json `
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer/special_tokens_map.json"

& $aria2 -c -d "$outdir\tokenizer" -o vocab.json `
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer/vocab.json"

& $aria2 -c -d "$outdir\tokenizer" -o merges.txt `
  "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/tokenizer/merges.txt"
