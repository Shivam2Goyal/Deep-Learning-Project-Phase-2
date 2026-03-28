"""
NOVEL CONTRIBUTION: Quantitative Inpainted Region Localization

The paper (Section 5.4, Fig 6) shows only qualitative heatmaps and explicitly states:
"In future work we aim to investigate how the reconstruction error can be 
used to predict the precise locations of inpainted regions."

We are the first to quantitatively evaluate this using:
  - IoU (Intersection over Union)
  - F1 Score (Dice coefficient)
  - PR-AUC (Precision-Recall Area Under Curve)

Three thresholding strategies are compared:
  1. Fixed Global Threshold (bottom percentile of heatmap)
  2. Otsu's Automatic Thresholding (finds optimal bimodal split)
  3. Adaptive Local Thresholding (spatially varying threshold)

Key: Inpainted regions have LOW reconstruction error (they lie on the AE manifold).
Real regions have HIGH reconstruction error.
"""

import argparse
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from PIL import Image
from sklearn.metrics import f1_score, average_precision_score
from tqdm import tqdm

from aeroblade.data import ImageFolder
from aeroblade.distances import LPIPS
from aeroblade.image import compute_reconstructions
from aeroblade.misc import safe_mkdir


# ----------------------------------------------------------------
# Heatmap computation
# ----------------------------------------------------------------

def compute_heatmap(inpainted_path: Path,
                    repo_id: str,
                    reconstruction_root: Path) -> np.ndarray:
    """
    Compute AEROBLADE spatial reconstruction error heatmap.
    Uses SD1.5 AE (matches SD inpainting model family).
    Computes ALL LPIPS layers and sums them — matches paper Fig 6 method.
    """
    ds_inp = ImageFolder([inpainted_path])

    rec_paths = compute_reconstructions(
        ds=ds_inp,
        repo_id=repo_id,
        output_root=reconstruction_root,
        seed=1,
        batch_size=1,
        num_workers=0,
    )
    ds_rec = ImageFolder(rec_paths)

    # Use ALL layers summed (lpips_vgg_0), spatial=True, no output_size
    # This matches what paper does in Fig 6 — omit spatial averaging
    lpips_metric = LPIPS(
        net="vgg",
        layer=0,        # layer 0 = sum of all layers
        spatial=True,   # keep spatial, no averaging
        output_size=512,
        batch_size=1,
        num_workers=0,
    )

    with torch.no_grad():
        dist_dict, _ = lpips_metric.compute(ds_inp, ds_rec)

    # Stored as negative — flip to positive
    # HIGH value = high reconstruction error = REAL region
    # LOW  value = low  reconstruction error = INPAINTED region  
    heatmap = -dist_dict["lpips_vgg_0"][0, 0].float().numpy()
    return heatmap

# ----------------------------------------------------------------
# Thresholding strategies
# ----------------------------------------------------------------

def threshold_fixed(heatmap: np.ndarray, percentile: float = 25.0) -> np.ndarray:
    """
    Fixed global threshold: lowest percentile = inpainted regions.
    Pixels with error BELOW threshold are predicted as inpainted.
    """
    thresh = np.percentile(heatmap, percentile)
    return (heatmap < thresh).astype(np.uint8)


def threshold_otsu(heatmap: np.ndarray) -> np.ndarray:
    """
    Otsu's automatic thresholding.
    Operates on inverted heatmap so inpainted (low error) = bright foreground.
    """
    # Invert: inpainted regions become HIGH (foreground)
    inverted = -heatmap
    h_min, h_max = inverted.min(), inverted.max()
    if h_max - h_min < 1e-8:
        return np.zeros_like(heatmap, dtype=np.uint8)

    # Normalize to uint8 for OpenCV
    norm = ((inverted - h_min) / (h_max - h_min) * 255).astype(np.uint8)
    _, mask = cv2.threshold(norm, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def threshold_adaptive(heatmap: np.ndarray, block_size: int = 51) -> np.ndarray:
    """
    Adaptive local thresholding.
    Uses local Gaussian-weighted mean — handles spatially varying illumination.
    Block size must be odd.
    """
    inverted = -heatmap
    h_min, h_max = inverted.min(), inverted.max()
    if h_max - h_min < 1e-8:
        return np.zeros_like(heatmap, dtype=np.uint8)

    norm = ((inverted - h_min) / (h_max - h_min) * 255).astype(np.uint8)

    if block_size % 2 == 0:
        block_size += 1

    mask = cv2.adaptiveThreshold(
        norm, 1,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=block_size,
        C=2,
    )
    return mask


# ----------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------

def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    intersection = np.logical_and(pred, gt).sum()
    union        = np.logical_or(pred, gt).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def compute_f1(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.flatten().astype(np.uint8)
    g = gt.flatten().astype(np.uint8)
    if g.sum() == 0 and p.sum() == 0:
        return 1.0
    if g.sum() == 0 or p.sum() == 0:
        return 0.0
    return float(f1_score(g, p, zero_division=0))


# ----------------------------------------------------------------
# Main
# ----------------------------------------------------------------

def main(args):
    safe_mkdir(args.output_dir)

    # --- Load dataset ---
    inpainted_paths = sorted((args.inpainting_dir / "inpainted").glob("*.png"))
    mask_paths      = sorted((args.inpainting_dir / "masks").glob("*.png"))
    orig_paths      = sorted((args.inpainting_dir / "original").glob("*.png"))

    assert len(inpainted_paths) == len(mask_paths) == len(orig_paths), \
        "Mismatch between inpainted/mask/original counts"

    n = len(inpainted_paths)
    print(f"Found {n} images. Computing heatmaps and evaluating localization...")

    # Storage
    results       = []
    heatmap_store = {}
    gt_mask_store = {}
    pred_store    = {}

    for inp_path, mask_path in tqdm(
        zip(inpainted_paths, mask_paths), total=n, desc="Evaluating"
    ):
        name = inp_path.stem

        # Ground truth mask
        gt_np     = np.array(Image.open(mask_path).convert("L"))
        gt_binary = (gt_np > 127).astype(np.uint8)

        # Compute heatmap
        try:
            heatmap = compute_heatmap(
                inpainted_path=inp_path,
                repo_id=args.repo_id,
                reconstruction_root=args.reconstruction_root,
            )
        except Exception as e:
            print(f"  Skipping {name}: {e}")
            continue

        # Three predictions
        pred_fixed    = threshold_fixed(heatmap, percentile=args.percentile)
        pred_otsu     = threshold_otsu(heatmap)
        pred_adaptive = threshold_adaptive(heatmap, block_size=args.adaptive_block_size)

        # Store
        heatmap_store[name] = heatmap
        gt_mask_store[name] = gt_binary
        pred_store[name]    = {
            "fixed":    pred_fixed,
            "otsu":     pred_otsu,
            "adaptive": pred_adaptive,
        }

        # Metrics
        row = {
            "name":         name,
            "iou_fixed":    compute_iou(pred_fixed,    gt_binary),
            "iou_otsu":     compute_iou(pred_otsu,     gt_binary),
            "iou_adaptive": compute_iou(pred_adaptive, gt_binary),
            "f1_fixed":     compute_f1(pred_fixed,     gt_binary),
            "f1_otsu":      compute_f1(pred_otsu,      gt_binary),
            "f1_adaptive":  compute_f1(pred_adaptive,  gt_binary),
            "pr_auc": average_precision_score(
                gt_binary.flatten(),
                (-heatmap).flatten(),   # inverted: inpainted = high score
            ),
        }
        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_dir / "localization_results.csv", index=False)

    # ----------------------------------------------------------------
    # Print summary table
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("QUANTITATIVE LOCALIZATION RESULTS")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'Mean IoU':>10} {'Mean F1':>10}")
    print("-"*45)
    for method, label in [("fixed", "Fixed (Global)"),
                           ("otsu",  "Otsu (Automatic)"),
                           ("adaptive", "Adaptive (Local)")]:
        miou = results_df[f"iou_{method}"].mean()
        mf1  = results_df[f"f1_{method}"].mean()
        print(f"{label:<25} {miou:>10.4f} {mf1:>10.4f}")
    print(f"\n{'PR-AUC (raw heatmap)':<25} {results_df['pr_auc'].mean():>10.4f}")
    print(f"{'='*60}")
    print(f"(Evaluated on {len(results_df)} images)")

    # ----------------------------------------------------------------
    # Visualization 1: Best examples grid
    # ----------------------------------------------------------------
    # Pick 4 best examples by Otsu IoU
    top4_names = results_df.nlargest(4, "iou_otsu")["name"].tolist()

    fig, axes = plt.subplots(4, 7, figsize=(26, 15))
    col_titles = [
        "Original", "Inpainted",
        "Ground Truth\nMask", "AEROBLADE\nHeatmap",
        "Fixed\nThreshold", "Otsu\nThreshold", "Adaptive\nThreshold",
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight="bold")

    for row_idx, name in enumerate(top4_names):
        orig_img = np.array(
            Image.open(args.inpainting_dir / "original"  / f"{name}.png").convert("RGB")
        )
        inp_img  = np.array(
            Image.open(args.inpainting_dir / "inpainted" / f"{name}.png").convert("RGB")
        )
        gt       = gt_mask_store[name]
        heatmap  = heatmap_store[name]
        preds    = pred_store[name]

        # Normalize heatmap for display
        # Low error (inpainted) → bright in display
        disp = -heatmap  # flip: inpainted = high = bright
        disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-8)

        row_data = results_df[results_df["name"] == name].iloc[0]
        iou_f = row_data["iou_fixed"]
        iou_o = row_data["iou_otsu"]
        iou_a = row_data["iou_adaptive"]

        panels = [
            (orig_img,         None,  ""),
            (inp_img,          None,  ""),
            (gt,               "gray", ""),
            (disp,             "hot",  "bright = low error\n= inpainted"),
            (preds["fixed"],    "gray", f"IoU = {iou_f:.3f}"),
            (preds["otsu"],     "gray", f"IoU = {iou_o:.3f}"),
            (preds["adaptive"], "gray", f"IoU = {iou_a:.3f}"),
        ]

        for col, (img_data, cmap, xlabel) in enumerate(panels):
            ax = axes[row_idx, col]
            if len(img_data.shape) == 3:
                ax.imshow(img_data)
            else:
                vmax = 1 if img_data.max() <= 1.01 else 255
                ax.imshow(img_data, cmap=cmap, vmin=0, vmax=vmax)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        axes[row_idx, 0].set_ylabel(f"Image {name}", fontsize=8, rotation=90)

    plt.suptitle(
        "Inpainted Region Localization — Top 4 Examples by IoU\n"
        "Bright heatmap = low reconstruction error = inpainted region",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    out_grid = args.output_dir / "localization_grid.png"
    plt.savefig(out_grid, dpi=120, bbox_inches="tight")
    print(f"\nSaved example grid to {out_grid}")
    plt.close()

    # ----------------------------------------------------------------
    # Visualization 2: Metrics bar chart
    # ----------------------------------------------------------------
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5))

    methods = ["Fixed\n(Global)", "Otsu\n(Auto)", "Adaptive\n(Local)"]
    colors  = ["steelblue", "seagreen", "darkorange"]
    keys_m  = ["fixed", "otsu", "adaptive"]

    for ax, metric, ylabel, title in [
        (axes2[0], "iou", "Mean IoU",     "Localization IoU\nby Thresholding Method"),
        (axes2[1], "f1",  "Mean F1",      "Localization F1\nby Thresholding Method"),
    ]:
        vals = [results_df[f"{metric}_{k}"].mean() for k in keys_m]
        bars = ax.bar(methods, vals, color=colors, zorder=2, width=0.5)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{v:.4f}", ha="center", fontweight="bold", fontsize=11)

    # PR-AUC per-image distribution
    axes2[2].hist(results_df["pr_auc"], bins=20, color="mediumpurple",
                  edgecolor="white", zorder=2)
    axes2[2].axvline(results_df["pr_auc"].mean(), color="red", lw=2,
                     label=f"Mean = {results_df['pr_auc'].mean():.4f}")
    axes2[2].set_xlabel("PR-AUC per image")
    axes2[2].set_ylabel("Count")
    axes2[2].set_title("PR-AUC Distribution\n(Raw Heatmap Score)")
    axes2[2].legend()
    axes2[2].grid(True, alpha=0.3)

    plt.suptitle("Quantitative Inpainted Region Localization — Summary Metrics",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_bar = args.output_dir / "localization_metrics.png"
    plt.savefig(out_bar, dpi=150, bbox_inches="tight")
    print(f"Saved metrics chart to {out_bar}")
    plt.close()

    print("\nDone!")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantitative Inpainted Region Localization with AEROBLADE"
    )
    parser.add_argument(
        "--inpainting-dir", type=Path, default=Path("data/inpainting"),
        help="Directory containing original/, inpainted/, masks/ subdirs"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("output/05/inpainting_localization")
    )
    parser.add_argument(
        "--repo-id", default="runwayml/stable-diffusion-v1-5",
        help="AE for reconstruction. Must match inpainting model family."
    )
    parser.add_argument(
        "--reconstruction-root", type=Path,
        default=Path("data/reconstructions"),
        help="Where to cache AE reconstructions"
    )
    parser.add_argument(
        "--percentile", type=float, default=25.0,
        help="Bottom percentile used for fixed threshold (default: 25)"
    )
    parser.add_argument(
        "--adaptive-block-size", type=int, default=51,
        help="Block size for adaptive thresholding, must be odd (default: 51)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())