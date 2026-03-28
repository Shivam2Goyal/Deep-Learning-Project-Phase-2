import os
import torch
from pathlib import Path
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

SAVE_DIR = Path("data/raw/generated/sd1-5")
NUM_IMAGES = 200
os.makedirs(SAVE_DIR, exist_ok=True)

# Diverse prompts covering simple and complex scenes
# (complexity range is important for our novel contribution)
PROMPTS = [
    "a photo of a cat sitting on a couch",
    "a beautiful mountain landscape at sunset",
    "a busy city street with people and cars",
    "a close-up photo of a red rose",
    "a dense forest with sunlight filtering through trees",
    "a simple white coffee mug on a wooden table",
    "a crowded marketplace with colorful fruits and vegetables",
    "a calm blue ocean with gentle waves",
    "a detailed portrait of an elderly man",
    "a snowy mountain peak under clear blue sky",
    "a photo of a dog running in a park",
    "an aerial view of a city at night with lights",
    "a plate of colorful sushi rolls",
    "a field of sunflowers on a bright day",
    "a modern kitchen interior with stainless steel appliances",
    "a waterfall surrounded by tropical vegetation",
    "a photo of a bicycle parked on a cobblestone street",
    "a dramatic thunderstorm over the ocean",
    "a cozy living room with a fireplace",
    "a macro photo of a butterfly on a flower",
]

print("Loading SD1.5 pipeline...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe = pipe.to("cuda")
pipe.set_progress_bar_config(disable=True)

count = 0
pbar = tqdm(total=NUM_IMAGES, desc="Generating SD images")

while count < NUM_IMAGES:
    for prompt in PROMPTS:
        if count >= NUM_IMAGES:
            break
        try:
            image = pipe(
                prompt,
                height=512,
                width=512,
                num_inference_steps=50,
                generator=torch.Generator("cuda").manual_seed(count),
            ).images[0]
            image.save(SAVE_DIR / f"{count:04d}.png")
            count += 1
            pbar.update(1)
        except Exception as e:
            print(f"Error generating image {count}: {e}")

pbar.close()
print(f"Saved {count} generated images to {SAVE_DIR}")