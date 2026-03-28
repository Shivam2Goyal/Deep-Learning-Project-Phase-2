import os
import random
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionInpaintPipeline

# ---- Config ----
REAL_DIR    = Path("data/raw/real")
OUTPUT_DIR  = Path("data/inpainting")
NUM_IMAGES  = 50
SIZE        = 512
SEED        = 42
# ----------------

ORIG_DIR     = OUTPUT_DIR / "original"
INPAINTED_DIR = OUTPUT_DIR / "inpainted"
MASK_DIR     = OUTPUT_DIR / "masks"

for d in [ORIG_DIR, INPAINTED_DIR, MASK_DIR]:
    os.makedirs(d, exist_ok=True)

random.seed(SEED)


def make_random_mask(size=512):
    """
    Random rectangular mask covering 15-35% of image area.
    White (255) = inpaint this region. Black (0) = keep.
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    for _ in range(10000):
        w = random.randint(int(size * 0.20), int(size * 0.50))
        h = random.randint(int(size * 0.20), int(size * 0.50))
        frac = (w * h) / (size * size)
        if 0.15 <= frac <= 0.35:
            break
    x = random.randint(0, size - w)
    y = random.randint(0, size - h)
    mask[y:y+h, x:x+w] = 255
    return Image.fromarray(mask)


print("Loading SD inpainting pipeline (~2GB download if not cached)...")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

real_images = sorted(REAL_DIR.glob("*.png"))[:NUM_IMAGES]
print(f"Processing {len(real_images)} real images...")

for i, img_path in enumerate(tqdm(real_images, desc="Inpainting")):
    stem = f"{i:04d}"
    img      = Image.open(img_path).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    mask_pil = make_random_mask(SIZE)

    with torch.inference_mode():
        result = pipe(
            prompt="",
            image=img,
            mask_image=mask_pil,
            height=SIZE,
            width=SIZE,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=torch.Generator("cuda").manual_seed(SEED + i),
        ).images[0]

    img.save(ORIG_DIR / f"{stem}.png")
    result.save(INPAINTED_DIR / f"{stem}.png")
    mask_pil.save(MASK_DIR / f"{stem}.png")

print(f"\nDone! Saved to {OUTPUT_DIR}")