from diffusers import StableDiffusionXLPipeline
import torch

# Path to your local SDXL Base model folder
model_path = r"D:\University\Personal Project\Capstoneflask\models\sdxl"

# Load pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,  # FP16 for GPU
    use_safetensors=True
).to("cuda")

# Optional: reduce VRAM usage if your GPU has <12GB
pipe.enable_attention_slicing()

# Test prompt
prompt = "A futuristic cityscape at sunset, cyberpunk style"

# Generate
image = pipe(prompt).images[0]

# Save result
image.save("sdxl_test.png")

print("✅ Image generated and saved as sdxl_test.png")
