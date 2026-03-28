import os
import zipfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import io

# ---- CHANGE THIS to wherever you downloaded val2017.zip ----
ZIP_PATH = Path(r"C:\Users\Hp\Downloads\val2017.zip")
# ------------------------------------------------------------

SAVE_DIR = Path("data/raw/real")
NUM_IMAGES = 200
SIZE = 512

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Reading from {ZIP_PATH}...")

with zipfile.ZipFile(ZIP_PATH, "r") as zf:
    # get all jpg files inside the zip
    all_files = [f for f in zf.namelist() if f.endswith(".jpg")]
    print(f"Found {len(all_files)} images in zip")

    count = 0
    for fname in tqdm(all_files, desc="Processing images"):
        if count >= NUM_IMAGES:
            break
        try:
            with zf.open(fname) as f:
                img = Image.open(io.BytesIO(f.read())).convert("RGB")

            # center crop to square then resize to 512x512
            w, h = img.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top  = (h - min_dim) // 2
            img  = img.crop((left, top, left + min_dim, top + min_dim))
            img  = img.resize((SIZE, SIZE), Image.LANCZOS)

            img.save(SAVE_DIR / f"{count:04d}.png")
            count += 1

        except Exception as e:
            print(f"Skipping {fname}: {e}")

print(f"\nDone. Saved {count} real images to {SAVE_DIR}")