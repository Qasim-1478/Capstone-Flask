import os
import io
import base64
from datetime import datetime
from typing import Optional

from flask import Flask, request, jsonify, render_template, send_from_directory


# Lazy-loaded global pipelines
sdxl_txt2img_pipe = None
sdxl_img2img_pipe = None
sd_upscale_pipe = None
sdxl_inpaint_pipe = None


def get_device_and_dtype():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", torch.float16
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32
    except Exception:
        # Fallback if torch not installed yet
        return "cpu", None


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


def load_sdxl_txt2img():
    global sdxl_txt2img_pipe
    if sdxl_txt2img_pipe is not None:
        return sdxl_txt2img_pipe

    import torch
    from diffusers import StableDiffusionXLPipeline

    device, dtype = get_device_and_dtype()
    # Prefer local directory if provided
    local_dir = os.environ.get("SDXL_LOCAL_DIR")
    model_id = local_dir if (local_dir and os.path.isdir(local_dir)) else os.environ.get("SDXL_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
    local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype if dtype is not None else None,
        use_safetensors=True,
        local_files_only=local_only,
    )

    if device == "cuda":
        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()
    elif device == "mps":
        pipe = pipe.to(device)
    else:
        pipe = pipe.to("cpu")

    pipe.enable_attention_slicing()
    sdxl_txt2img_pipe = pipe
    return pipe


def load_sdxl_img2img():
    global sdxl_img2img_pipe
    if sdxl_img2img_pipe is not None:
        return sdxl_img2img_pipe

    import torch
    from diffusers import StableDiffusionXLImg2ImgPipeline

    device, dtype = get_device_and_dtype()
    local_dir = os.environ.get("SDXL_LOCAL_DIR")
    model_id = local_dir if (local_dir and os.path.isdir(local_dir)) else os.environ.get("SDXL_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
    local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype if dtype is not None else None,
        use_safetensors=True,
        local_files_only=local_only,
    )

    if device == "cuda":
        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()
    elif device == "mps":
        pipe = pipe.to(device)
    else:
        pipe = pipe.to("cpu")

    pipe.enable_attention_slicing()
    sdxl_img2img_pipe = pipe
    return pipe


def load_upscaler():
    global sd_upscale_pipe
    if sd_upscale_pipe is not None:
        return sd_upscale_pipe

    import torch
    from diffusers import StableDiffusionUpscalePipeline

    device, dtype = get_device_and_dtype()
    local_dir = os.environ.get("SD_UPSCALE_LOCAL_DIR")
    model_id = local_dir if (local_dir and os.path.isdir(local_dir)) else os.environ.get("SD_UPSCALE_MODEL_ID", "stabilityai/stable-diffusion-x4-upscaler")
    local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        torch_dtype=dtype if dtype is not None else None,
        use_safetensors=True,
        local_files_only=local_only,
    )

    if device == "cuda":
        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()
    elif device == "mps":
        pipe = pipe.to(device)
    else:
        pipe = pipe.to("cpu")

    pipe.enable_attention_slicing()
    sd_upscale_pipe = pipe
    return pipe


def load_sdxl_inpaint():
    global sdxl_inpaint_pipe
    if sdxl_inpaint_pipe is not None:
        return sdxl_inpaint_pipe

    import torch
    from diffusers import StableDiffusionXLInpaintPipeline

    device, dtype = get_device_and_dtype()
    local_dir = os.environ.get("SDXL_LOCAL_DIR")
    model_id = local_dir if (local_dir and os.path.isdir(local_dir)) else os.environ.get("SDXL_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
    local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype if dtype is not None else None,
        use_safetensors=True,
        local_files_only=local_only,
    )

    if device == "cuda":
        pipe = pipe.to(device)
        pipe.enable_xformers_memory_efficient_attention()
    elif device == "mps":
        pipe = pipe.to(device)
    else:
        pipe = pipe.to("cpu")

    pipe.enable_attention_slicing()
    sdxl_inpaint_pipe = pipe
    return pipe


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["UPLOAD_FOLDER"] = os.path.join(app.root_path, "static", "uploads")
    app.config["OUTPUT_FOLDER"] = os.path.join(app.root_path, "static", "outputs")
    ensure_output_dir(app.config["UPLOAD_FOLDER"]) 
    ensure_output_dir(app.config["OUTPUT_FOLDER"]) 

    @app.route("/")
    def index():
        return render_template("index.html")

    # Feature pages
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

    @app.post("/api/text2img")
    def api_text2img():
        try:
            data = request.form if request.form else request.json
            prompt = (data.get("prompt") if data else None) or request.form.get("prompt")
            negative_prompt = (data.get("negative_prompt") if data else None) or request.form.get("negative_prompt")
            steps = int(((data or {}).get("steps") or request.form.get("steps") or 30))
            guidance = float(((data or {}).get("guidance_scale") or request.form.get("guidance_scale") or 7.0))
            width = int(((data or {}).get("width") or request.form.get("width") or 1024))
            height = int(((data or {}).get("height") or request.form.get("height") or 1024))

            if not prompt or str(prompt).strip() == "":
                return jsonify({"error": "Missing prompt"}), 400

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
            return jsonify({"filename": filename, "url": f"/static/outputs/{filename}"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/api/img2img")
    def api_img2img():
        try:
            prompt = request.form.get("prompt")
            negative_prompt = request.form.get("negative_prompt")
            strength = float(request.form.get("strength", 0.7))
            steps = int(request.form.get("steps", 30))
            guidance = float(request.form.get("guidance_scale", 7.0))
            init_image_file = request.files.get("image")
            if not init_image_file:
                return jsonify({"error": "Missing image file"}), 400
            if not prompt or str(prompt).strip() == "":
                return jsonify({"error": "Missing prompt"}), 400

            # Load image
            from PIL import Image
            init_image = Image.open(init_image_file.stream).convert("RGB")

            pipe = load_sdxl_img2img()
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )
            image = result.images[0]
            filename = save_pil_image(image, app.config["OUTPUT_FOLDER"], "i2i")
            return jsonify({"filename": filename, "url": f"/static/outputs/{filename}"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/api/upscale")
    def api_upscale():
        try:
            prompt = request.form.get("prompt", None)  # optional, some upscalers accept it
            image_file = request.files.get("image")
            if not image_file:
                return jsonify({"error": "Missing image file"}), 400

            from PIL import Image
            lowres = Image.open(image_file.stream).convert("RGB")

            pipe = load_upscaler()
            # Some SD upscalers accept a prompt; if None, pass empty string
            result = pipe(
                prompt=prompt or "high detail, sharp, clean",
                image=lowres,
            )
            image = result.images[0]
            filename = save_pil_image(image, app.config["OUTPUT_FOLDER"], "upscale")
            return jsonify({"filename": filename, "url": f"/static/outputs/{filename}"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/api/denoise")
    def api_denoise():
        try:
            img_file = request.files.get("image")
            if not img_file:
                return jsonify({"error": "Missing image file"}), 400
            strength = int(request.form.get("strength", 20))
            strength = max(0, min(100, strength))

            from PIL import Image
            import numpy as np
            import cv2

            pil_img = Image.open(img_file.stream).convert("RGB")
            img = np.array(pil_img)[:, :, ::-1]  # RGB->BGR

            h = max(1, int(strength))
            hColor = h
            templateWindowSize = 7
            searchWindowSize = 21
            denoised = cv2.fastNlMeansDenoisingColored(img, None, h, hColor, templateWindowSize, searchWindowSize)
            denoised_rgb = denoised[:, :, ::-1]
            out_img = Image.fromarray(denoised_rgb)
            filename = save_pil_image(out_img, app.config["OUTPUT_FOLDER"], "denoise")
            return jsonify({"filename": filename, "url": f"/static/outputs/{filename}"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/api/inpaint")
    def api_inpaint():
        try:
            img_file = request.files.get("image")
            mask_file = request.files.get("mask")
            prompt = request.form.get("prompt")
            negative_prompt = request.form.get("negative_prompt")
            steps = int(request.form.get("steps", 30))
            guidance = float(request.form.get("guidance_scale", 7.0))

            if not img_file or not mask_file:
                return jsonify({"error": "Missing image or mask file"}), 400
            if not prompt or str(prompt).strip() == "":
                return jsonify({"error": "Missing prompt"}), 400

            from PIL import Image
            init_image = Image.open(img_file.stream).convert("RGB")
            mask_image = Image.open(mask_file.stream).convert("L")  # white=mask, black=keep

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
            return jsonify({"filename": filename, "url": f"/static/outputs/{filename}"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return app


if __name__ == "__main__":
    app = create_app()
    #host = os.environ.get("FLASK_HOST", "0.0.0.0")
    #port = int(os.environ.get("FLASK_PORT", "5000"))
    #debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=True)


