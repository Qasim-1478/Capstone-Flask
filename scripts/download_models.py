import os
from pathlib import Path


def main():
    base_model = os.environ.get("SDXL_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
    upscaler_model = os.environ.get("SD_UPSCALE_MODEL_ID", "stabilityai/stable-diffusion-x4-upscaler")
    out_root = Path(os.environ.get("MODELS_DIR", "models")).resolve()
    sdxl_out = out_root / "sdxl"
    up_out = out_root / "x4-upscaler"
    sdxl_out.mkdir(parents=True, exist_ok=True)
    up_out.mkdir(parents=True, exist_ok=True)

    print(f"Downloading SDXL base to {sdxl_out} ...")
    from diffusers import StableDiffusionXLPipeline
    StableDiffusionXLPipeline.from_pretrained(base_model, use_safetensors=True).save_pretrained(str(sdxl_out))

    print(f"Downloading SD x4 upscaler to {up_out} ...")
    from diffusers import StableDiffusionUpscalePipeline
    StableDiffusionUpscalePipeline.from_pretrained(upscaler_model, use_safetensors=True).save_pretrained(str(up_out))

    print("Done. Set SDXL_LOCAL_DIR and SD_UPSCALE_LOCAL_DIR to these folders for offline use.")


if __name__ == "__main__":
    main()


