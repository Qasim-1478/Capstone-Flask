import os
import io
import base64
import torch
import gc

from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionUpscalePipeline,
    StableDiffusionXLInpaintPipeline
)

from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import numpy as np
import cv2


# ============================================================
# ✅ GLOBAL PIPELINES (will load/unload automatically)
# ============================================================
sdxl_txt2img_pipe = None
sdxl_img2img_pipe = None
sd_upscale_pipe = None
sdxl_inpaint_pipe = None


# ============================================================
# ✅ VRAM CLEANER – AUTO UNLOAD EVERYTHING
# ============================================================
def unload_all_pipelines():
    global sdxl_txt2img_pipe, sdxl_img2img_pipe, sd_upscale_pipe, sdxl_inpaint_pipe

    sdxl_txt2img_pipe = None
    sdxl_img2img_pipe = None
    sd_upscale_pipe = None
    sdxl_inpaint_pipe = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ============================================================
# ✅ DEVICE + DTYPE
# ============================================================
def get_device_and_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


# ============================================================
# ✅ UTILITIES
# ============================================================
def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_pil_image(pil_image, out_dir: str, prefix: str) -> str:
    ensure_output_dir(out_dir)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{timestamp}.png"
    out_path = os.path.join(out_dir, filename)
    pil_image.save(out_path)
    return filename


def pil_to_base64(pil_image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ============================================================
# ✅ VRAM-OPTIMIZED LOADERS (Auto reload + Low-memory setup)
# ============================================================

def load_sdxl_txt2img():
    global sdxl_txt2img_pipe
    if sdxl_txt2img_pipe is not None:
        return sdxl_txt2img_pipe

    device, dtype = get_device_and_dtype()
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    if device == "cuda":
        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing()

    sdxl_txt2img_pipe = pipe
    return pipe


def load_sdxl_img2img():
    global sdxl_img2img_pipe
    if sdxl_img2img_pipe is not None:
        return sdxl_img2img_pipe

    device, dtype = get_device_and_dtype()
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    if device == "cuda":
        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing()

    sdxl_img2img_pipe = pipe
    return pipe


def load_upscaler():
    global sd_upscale_pipe
    if sd_upscale_pipe is not None:
        return sd_upscale_pipe

    device, dtype = get_device_and_dtype()
    model_id = "stabilityai/stable-diffusion-x4-upscaler"

    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    if device == "cuda":
        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing()

    sd_upscale_pipe = pipe
    return pipe


def load_sdxl_inpaint():
    global sdxl_inpaint_pipe
    if sdxl_inpaint_pipe is not None:
        return sdxl_inpaint_pipe

    device, dtype = get_device_and_dtype()
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    if device == "cuda":
        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_vae_slicing()
    pipe.enable_vae_tiling()
    pipe.enable_sequential_cpu_offload()
    pipe.enable_attention_slicing()

    sdxl_inpaint_pipe = pipe
    return pipe


# ============================================================
# ✅ FLASK APP
# ============================================================
def create_app() -> Flask:
    app = Flask(__name__)
    app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "static", "uploads")
    app.config["OUTPUT_FOLDER"] = os.path.join(app.root_path, "static", "outputs")

    ensure_output_dir(app.config["UPLOAD_FOLDER"])
    ensure_output_dir(app.config["OUTPUT_FOLDER"])

    # ------------------------------
    # ✅ Render pages
    # ------------------------------
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/text2img")
    def page_text2img():
        return render_template("text2img.html")

    @app.route("/img2img")
    def page_img2img():
        return render_template("img2img.html")

    @app.route("/upscale")
    def page_upscale():
        return render_template("upscale.html")

    @app.route("/denoise")
    def page_denoise():
        return render_template("denoise.html")

    @app.route("/inpaint")
    def page_inpaint():
        return render_template("inpaint.html")

    @app.route("/static/outputs/<path:filename>")
    def serve_output(filename: str):
        return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

    # =====================================================
    # ✅ TEXT 2 IMG API (Auto unload)
    # =====================================================
    @app.post("/api/text2img")
    def api_text2img():
        try:
            data = request.form if request.form else request.json

            prompt = data.get("prompt")
            negative_prompt = data.get("negative_prompt")
            steps = int(data.get("steps", 30))
            guidance = float(data.get("guidance_scale", 7.0))
            width = int(data.get("width", 1024))
            height = int(data.get("height", 1024))

            pipe = load_sdxl_txt2img()
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
            )

            image = result.images[0]
            filename = save_pil_image(image, app.config["OUTPUT_FOLDER"], "t2i")

            unload_all_pipelines()
            return jsonify({"filename": filename, "url": f"/static/outputs/{filename}"})

        except Exception as e:
            unload_all_pipelines()
            return jsonify({"error": str(e)}), 500

    # =====================================================
    # ✅ IMG2IMG API (Auto unload)
    # =====================================================
    @app.post("/api/img2img")
    def api_img2img():
        try:
            prompt = request.form.get("prompt")
            negative_prompt = request.form.get("negative_prompt")
            strength = float(request.form.get("strength", 0.7))
            steps = int(request.form.get("steps", 30))
            guidance = float(request.form.get("guidance_scale", 7.0))

            init_file = request.files.get("image")
            init_image = Image.open(init_file.stream).convert("RGB")

            pipe = load_sdxl_img2img()
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance
            )

            image = result.images[0]
            filename = save_pil_image(image, app.config["OUTPUT_FOLDER"], "i2i")

            unload_all_pipelines()
            return jsonify({"filename": filename, "url": f"/static/outputs/{filename}"})
        except Exception as e:
            unload_all_pipelines()
            return jsonify({"error": str(e)}), 500

    # =====================================================
    # ✅ UPSCALE API (Auto unload)
    # =====================================================
    @app.post("/api/upscale")
    def api_upscale():
        try:
            prompt = request.form.get("prompt", "high detail, sharp, clean")
            img_file = request.files.get("image")
            lowres = Image.open(img_file.stream).convert("RGB")

            pipe = load_upscaler()
            result = pipe(prompt=prompt, image=lowres)

            image = result.images[0]
            filename = save_pil_image(image, app.config["OUTPUT_FOLDER"], "upscale")

            unload_all_pipelines()
            return jsonify({"filename": filename, "url": f"/static/outputs/{filename}"})
        except Exception as e:
            unload_all_pipelines()
            return jsonify({"error": str(e)}), 500

    # =====================================================
    # ✅ DENOISE API (No SDXL used)
    # =====================================================
    @app.post("/api/denoise")
    def api_denoise():
        try:
            img_file = request.files.get("image")
            strength = int(request.form.get("strength", 20))

            pil_img = Image.open(img_file.stream).convert("RGB")
            img = np.array(pil_img)[:, :, ::-1]

            denoised = cv2.fastNlMeansDenoisingColored(
                img, None,
                strength, strength,
                7, 21
            )
            denoised = denoised[:, :, ::-1]
            out_img = Image.fromarray(denoised)

            filename = save_pil_image(out_img, app.config["OUTPUT_FOLDER"], "denoise")
            return jsonify({"filename": filename, "url": f"/static/outputs/{filename}"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # =====================================================
    # ✅ INPAINT API (Auto unload)
    # =====================================================
    @app.post("/api/inpaint")
    def api_inpaint():
        try:
            img_file = request.files.get("image")
            mask_file = request.files.get("mask")
            prompt = request.form.get("prompt")
            negative_prompt = request.form.get("negative_prompt")
            steps = int(request.form.get("steps", 30))
            guidance = float(request.form.get("guidance_scale", 7.0))

            init_image = Image.open(img_file.stream).convert("RGB")
            mask_image = Image.open(mask_file.stream).convert("L")

            pipe = load_sdxl_inpaint()
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                mask_image=mask_image,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )

            image = result.images[0]
            filename = save_pil_image(image, app.config["OUTPUT_FOLDER"], "inpaint")

            unload_all_pipelines()
            return jsonify({"filename": filename, "url": f"/static/outputs/{filename}"})
        except Exception as e:
            unload_all_pipelines()
            return jsonify({"error": str(e)}), 500

    return app


# ============================================================
# ✅ RUN SERVER
# ============================================================
if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
