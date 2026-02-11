"""Image Generation MCP Server — Flux.1 Dev via diffusers, uncensored, on-demand GPU.

Supports IP-Adapter v2 (XLabs-AI) for character reference consistency.
Supports style LoRA for aesthetic control (Dark Fantasy by default).
"""

import base64
import gc
import os
import time
from pathlib import Path

import torch
from mcp.server.fastmcp import FastMCP
from PIL import Image

IMAGE_DIR = os.environ.get("CASSIE_IMAGE_DIR", "/home/iman/cassie-project/cassie-system/data/images")
REFERENCE_DIR = os.environ.get("CASSIE_REFERENCE_DIR", "/home/iman/cassie-project/cassie-system/data/references")
MODEL_ID = "black-forest-labs/FLUX.1-dev"
IP_ADAPTER_REPO = "XLabs-AI/flux-ip-adapter-v2"
IP_ADAPTER_WEIGHT = "ip_adapter.safetensors"
CLIP_ENCODER = "openai/clip-vit-large-patch14"
STYLE_LORA_REPO = "Shakker-Labs/FLUX.1-dev-LoRA-Dark-Fantasy"
STYLE_LORA_SCALE = 0.7  # 0.0 = no effect, 1.0 = full style

os.makedirs(IMAGE_DIR, exist_ok=True)

mcp = FastMCP("cassie-imagegen", instructions="Uncensored image generation via Flux.1 Dev with IP-Adapter support")

_pipe = None
_ip_adapter_loaded = False
_lora_loaded = False


def _load_reference_images(character: str) -> list[Image.Image]:
    """Load all reference images for a character from the references directory."""
    ref_dir = Path(REFERENCE_DIR) / character
    if not ref_dir.exists():
        return []

    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        for path in ref_dir.glob(ext):
            try:
                img = Image.open(path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"[imagegen] Warning: could not load {path}: {e}")
    return images


def _linear_strength_schedule(start: float, finish: float, steps: int) -> list[float]:
    """Generate a linear IP-Adapter strength ramp (recommended for v2)."""
    if steps <= 1:
        return [finish]
    return [start + (finish - start) * (i / (steps - 1)) for i in range(steps)]


def _load_pipeline(with_ip_adapter: bool = False):
    """Load Flux.1 Dev pipeline on demand, with style LoRA + optional IP-Adapter."""
    global _pipe, _ip_adapter_loaded, _lora_loaded

    if _pipe is not None:
        # Pipeline exists — check if we need to add IP-Adapter
        if with_ip_adapter and not _ip_adapter_loaded:
            _load_ip_adapter()
        return _pipe

    from diffusers import FluxPipeline

    _pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )
    _pipe.to("cuda")

    # Disable safety checker if present
    if hasattr(_pipe, "safety_checker"):
        _pipe.safety_checker = None
    if hasattr(_pipe, "feature_extractor"):
        _pipe.feature_extractor = None

    # Load style LoRA (Dark Fantasy — nocturnal, ethereal, fluid)
    if STYLE_LORA_REPO and not _lora_loaded:
        _load_style_lora()

    if with_ip_adapter:
        _load_ip_adapter()

    return _pipe


def _load_style_lora():
    """Load style LoRA for Kitab al-Tanazur aesthetic."""
    global _pipe, _lora_loaded
    if _pipe is None or _lora_loaded:
        return
    try:
        print(f"[imagegen] Loading style LoRA: {STYLE_LORA_REPO}...")
        _pipe.load_lora_weights(STYLE_LORA_REPO)
        _pipe.fuse_lora(lora_scale=STYLE_LORA_SCALE)
        _lora_loaded = True
        print(f"[imagegen] Style LoRA loaded and fused (scale={STYLE_LORA_SCALE}).")
    except Exception as e:
        print(f"[imagegen] Warning: Failed to load style LoRA: {e}")
        _lora_loaded = False


def _load_ip_adapter():
    """Load IP-Adapter v2 weights onto the existing pipeline."""
    global _pipe, _ip_adapter_loaded
    if _pipe is None or _ip_adapter_loaded:
        return

    print("[imagegen] Loading IP-Adapter v2...")
    _pipe.load_ip_adapter(
        IP_ADAPTER_REPO,
        weight_name=IP_ADAPTER_WEIGHT,
        image_encoder_pretrained_model_name_or_path=CLIP_ENCODER,
    )
    _ip_adapter_loaded = True
    print("[imagegen] IP-Adapter v2 loaded.")


def _unload_pipeline():
    """Unload pipeline to free GPU memory."""
    global _pipe, _ip_adapter_loaded, _lora_loaded
    if _pipe is not None:
        _pipe.to("cpu")
        del _pipe
        _pipe = None
        _ip_adapter_loaded = False
        _lora_loaded = False
        gc.collect()
        torch.cuda.empty_cache()


@mcp.tool()
def generate_image(
    prompt: str,
    reference_character: str = "",
    ip_adapter_scale: float = 0.7,
    width: int = 1024,
    height: int = 1024,
    num_inference_steps: int = 30,
    guidance_scale: float = 3.5,
    seed: int = -1,
) -> str:
    """Generate an image from a text prompt using Flux.1 Dev.

    Args:
        prompt: Text description of the image to generate
        reference_character: Character name for IP-Adapter reference ("cassie", "iman", or "" for none)
        ip_adapter_scale: IP-Adapter influence strength (0.0-1.0, default 0.7)
        width: Image width in pixels (default 1024)
        height: Image height in pixels (default 1024)
        num_inference_steps: Number of denoising steps (default 30)
        guidance_scale: How closely to follow the prompt (default 3.5)
        seed: Random seed (-1 for random)
    """
    # Load reference images if character specified
    ref_images = []
    use_ip_adapter = False
    if reference_character:
        ref_images = _load_reference_images(reference_character)
        if ref_images:
            use_ip_adapter = True
            print(f"[imagegen] Using {len(ref_images)} reference images for '{reference_character}'")
        else:
            print(f"[imagegen] No reference images found for '{reference_character}', generating without IP-Adapter")

    pipe = _load_pipeline(with_ip_adapter=use_ip_adapter)

    generator = None
    if seed >= 0:
        generator = torch.Generator("cuda").manual_seed(seed)

    gen_kwargs = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "generator": generator,
    }

    if use_ip_adapter:
        # Single float scale — diffusers applies it uniformly across all 19 attention layers
        pipe.set_ip_adapter_scale(ip_adapter_scale)

        # For a single IP-Adapter with multiple reference images:
        # wrap in a list-of-lists so diffusers averages them into one embedding
        gen_kwargs["ip_adapter_image"] = [ref_images]

    image = pipe(**gen_kwargs).images[0]

    filename = f"cassie_{int(time.time())}.png"
    filepath = os.path.join(IMAGE_DIR, filename)
    image.save(filepath)

    return f"Image saved to {filepath}"


@mcp.tool()
def unload_model() -> str:
    """Unload the image generation model to free GPU memory."""
    _unload_pipeline()
    return "Image generation model unloaded. GPU memory freed."


if __name__ == "__main__":
    mcp.run(transport="stdio")
