"""
NOVEL CONTRIBUTION: Complexity-Normalized AEROBLADE — Robustness to JPEG Compression

Key insight from paper's Fig 5 (never exploited algorithmically):
  Real images:      reconstruction error CORRELATES with complexity
  Generated images: this correlation DISAPPEARS

Our contribution:
  Fit expected_error = f(complexity) on CLEAN real images.
  Define normalized_score = actual_error / expected_error(complexity)

  This makes the score robust to JPEG compression because:
  JPEG compression reduces image complexity AND reconstruction error together.
  Our normalization accounts for this joint reduction, maintaining
  the real/generated separation even after compression.

We show: Original AEROBLADE degrades under JPEG compression.
         Complexity-Normalized AEROBLADE is more robust.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

from aeroblade.complexities import JPEG
from aeroblade.data import ImageFolder
from aeroblade.misc import safe_mkdir


def compute_image_complexity(image_dir: Path, amount: int = None) -> pd.DataFrame:
    """Compute per-image JPEG complexity (quality=50, whole image)."""
    ds = ImageFolder(image_dir, amount=amount)
    jpeg = JPEG(quality=50, patch_size=None, patch_stride=None)
    comp_dict, files = jpeg.compute(ds)
    complexities = comp_dict["jpeg_50"].squeeze().numpy().astype(np.float64)
    return pd.DataFrame({
        "file": files,
        "complexity": complexities,
        "dir": str(image_dir),
    })


def main(args):
    safe_mkdir(args.output_dir)

    # ----------------------------------------------------------------
    # 1. Load distances
    # ----------------------------------------------------------------
    print(f"Loading distances from {args.distances_parquet}...")
    distances = pd.read_parquet(args.distances_parquet)

    distances["repo_id"]         = distances["repo_id"].astype(str)
    distances["distance_metric"] = distances["distance_metric"].astype(str)
    distances["transform"]       = distances["transform"].astype(str)
    distances["dir"]             = distances["dir"].astype(str)

    print(f"repo_ids   : {distances['repo_id'].unique().tolist()}")
    print(f"transforms : {distances['transform'].unique().tolist()}")
    print(f"metrics    : {distances['distance_metric'].unique().tolist()}")

    available_repos = distances["repo_id"].unique().tolist()
    repo_id = "max" if "max" in available_repos else \
              [r for r in available_repos if r != "max"][0]
    print(f"\nUsing repo_id: '{repo_id}'")

    dist_df = distances[
        (distances["repo_id"] == repo_id) &
        (distances["distance_metric"] == args.distance_metric)
    ].copy()

    dist_df["distance_pos"] = -dist_df["distance"].astype(np.float64)
    dist_df["is_real"] = dist_df["dir"] == str(args.real_dir)

    print(f"\nTransforms available: {dist_df['transform'].unique().tolist()}")
    print(f"Total records: {len(dist_df)}")

    # ----------------------------------------------------------------
    # 2. Compute complexity for all image directories
    # ----------------------------------------------------------------
    print("\nComputing JPEG complexity on all image directories...")
    all_dirs = [Path(d) for d in dist_df["dir"].unique()]
    complexity_dfs = []
    for d in all_dirs:
        print(f"  {d}")
        complexity_dfs.append(compute_image_complexity(d, amount=args.amount))
    complexity_df = pd.concat(complexity_dfs, ignore_index=True)

    # ----------------------------------------------------------------
    # 3. Fit normalization curve on CLEAN real images only
    # ----------------------------------------------------------------
    clean_real = dist_df[
        (dist_df["transform"] == "clean") & dist_df["is_real"]
    ].merge(complexity_df, on=["file", "dir"], how="inner")
    clean_real = clean_real[
        np.isfinite(clean_real["distance_pos"]) &
        np.isfinite(clean_real["complexity"])
    ].copy()

    # Cast to float64 explicitly
    X_fit = clean_real["complexity"].values.astype(np.float64).reshape(-1, 1)
    y_fit = clean_real["distance_pos"].values.astype(np.float64)

    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X_fit, y_fit)
    r2 = model.score(X_fit, y_fit)
    print(f"\nCurve fit on {len(X_fit)} clean real images | R²={r2:.4f}")

    # Safe x range for plotting — derived from float64 clean data
    x_plot_min = float(X_fit.min())
    x_plot_max = float(X_fit.max())

    # ----------------------------------------------------------------
    # 4. For each transform level, compute AP and normalized AP
    # ----------------------------------------------------------------
    transform_order = ["clean", "jpeg_80", "jpeg_70", "jpeg_60", "jpeg_50"]
    transform_labels = {
        "clean":   "Clean\n(q=100)",
        "jpeg_80": "JPEG\n(q=80)",
        "jpeg_70": "JPEG\n(q=70)",
        "jpeg_60": "JPEG\n(q=60)",
        "jpeg_50": "JPEG\n(q=50)",
    }

    results = []
    available_transforms = dist_df["transform"].unique().tolist()

    for transform in transform_order:
        if transform not in available_transforms:
            print(f"  Skipping {transform} (not in data)")
            continue

        t_df = dist_df[dist_df["transform"] == transform].copy()
        t_df = t_df.merge(complexity_df, on=["file", "dir"], how="inner")
        t_df = t_df[
            np.isfinite(t_df["distance_pos"]) &
            np.isfinite(t_df["complexity"])
        ].copy()

        y_true     = t_df["is_real"].astype(int).values
        raw_scores = t_df["distance_pos"].values.astype(np.float64)

        X_t      = t_df["complexity"].values.astype(np.float64).reshape(-1, 1)
        expected = np.clip(model.predict(X_t), 1e-6, None)
        norm_scores = raw_scores / expected

        ap_orig = average_precision_score(y_true=y_true, y_score=raw_scores)
        ap_norm = average_precision_score(y_true=y_true, y_score=norm_scores)

        mean_real = t_df[t_df["is_real"]]["distance_pos"].mean()
        mean_gen  = t_df[~t_df["is_real"]]["distance_pos"].mean()
        std_pool  = t_df["distance_pos"].std() + 1e-9
        cohen_d_orig = (mean_real - mean_gen) / std_pool

        X_real = t_df[t_df["is_real"]]["complexity"].values.astype(np.float64).reshape(-1,1)
        X_gen  = t_df[~t_df["is_real"]]["complexity"].values.astype(np.float64).reshape(-1,1)
        norm_real = (t_df[t_df["is_real"]]["distance_pos"].values.astype(np.float64) /
                     np.clip(model.predict(X_real), 1e-6, None))
        norm_gen  = (t_df[~t_df["is_real"]]["distance_pos"].values.astype(np.float64) /
                     np.clip(model.predict(X_gen),  1e-6, None))
        std_pool_n   = norm_scores.std() + 1e-9
        cohen_d_norm = (norm_real.mean() - norm_gen.mean()) / std_pool_n

        results.append({
            "transform":     transform,
            "label":         transform_labels.get(transform, transform),
            "ap_original":   ap_orig,
            "ap_normalized": ap_norm,
            "improvement":   ap_norm - ap_orig,
            "cohen_d_orig":  cohen_d_orig,
            "cohen_d_norm":  cohen_d_norm,
        })

        print(f"  {transform:12s} | AP_orig={ap_orig:.4f} | AP_norm={ap_norm:.4f} "
              f"| Δ={ap_norm-ap_orig:+.4f} | Cohen's d: {cohen_d_orig:.2f}→{cohen_d_norm:.2f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_dir / "robustness_results.csv", index=False)

    # ----------------------------------------------------------------
    # 5. Visualizations
    # ----------------------------------------------------------------
    fig = plt.figure(figsize=(20, 10))

    # --- Plot 1: Fig 5 reproduction ---
    ax1 = fig.add_subplot(2, 3, 1)
    clean_all = dist_df[dist_df["transform"] == "clean"].merge(
        complexity_df, on=["file", "dir"], how="inner"
    )
    clean_all = clean_all[
        np.isfinite(clean_all["distance_pos"]) &
        np.isfinite(clean_all["complexity"])
    ].copy()

    real_p = clean_all[clean_all["is_real"]]
    gen_p  = clean_all[~clean_all["is_real"]]

    ax1.scatter(real_p["complexity"].astype(float), real_p["distance_pos"],
                alpha=0.5, s=15, color="steelblue", label="Real", zorder=2)
    ax1.scatter(gen_p["complexity"].astype(float),  gen_p["distance_pos"],
                alpha=0.5, s=15, color="tomato",    label="Generated", zorder=2)

    x_line = np.linspace(x_plot_min, x_plot_max, 200).reshape(-1, 1).astype(np.float64)
    y_line = model.predict(x_line)
    ax1.plot(x_line.flatten(), y_line, color="navy", lw=2.5,
             linestyle="--", label=f"Fitted curve (R²={r2:.3f})", zorder=3)

    ax1.set_xlabel("JPEG Complexity (quality=50)")
    ax1.set_ylabel("Reconstruction Error (LPIPS)")
    ax1.set_title("Fig 5 Reproduction:\nComplexity vs Reconstruction Error")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: AP robustness comparison ---
    ax2 = fig.add_subplot(2, 3, 2)
    labels = results_df["label"].tolist()
    x_pos  = np.arange(len(labels))
    width  = 0.35
    bars1 = ax2.bar(x_pos - width/2, results_df["ap_original"],  width,
                    color="steelblue", label="Original AEROBLADE", zorder=2)
    bars2 = ax2.bar(x_pos + width/2, results_df["ap_normalized"], width,
                    color="seagreen",  label="Complexity-Normalized", zorder=2)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels, fontsize=9)
    ymin = max(0, results_df[["ap_original","ap_normalized"]].min().min() - 0.1)
    ax2.set_ylim(ymin, 1.05)
    ax2.set_ylabel("Average Precision (AP)")
    ax2.set_title("Robustness to JPEG Compression:\nOriginal vs Normalized")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")
    for bar in bars1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{bar.get_height():.4f}", ha="center", va="bottom",
                 fontsize=7, color="steelblue", fontweight="bold")
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{bar.get_height():.4f}", ha="center", va="bottom",
                 fontsize=7, color="seagreen", fontweight="bold")

    # --- Plot 3: Cohen's d ---
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(results_df["label"], results_df["cohen_d_orig"],
             "o-", color="steelblue", lw=2, ms=8, label="Original AEROBLADE")
    ax3.plot(results_df["label"], results_df["cohen_d_norm"],
             "s-", color="seagreen",  lw=2, ms=8, label="Complexity-Normalized")
    ax3.fill_between(range(len(labels)),
                     results_df["cohen_d_orig"].values,
                     results_df["cohen_d_norm"].values,
                     alpha=0.15, color="seagreen", label="Normalization gain")
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, fontsize=9)
    ax3.set_ylabel("Cohen's d (score separability)")
    ax3.set_title("Score Separability\n(higher = better separated)")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # --- Plots 4-6: Score distributions ---
    transform_subset = [t for t in transform_order if t in available_transforms][:3]
    for idx, transform in enumerate(transform_subset):
        ax = fig.add_subplot(2, 3, 4 + idx)
        t_df = dist_df[dist_df["transform"] == transform].copy()
        t_df = t_df.merge(complexity_df, on=["file", "dir"], how="inner")
        t_df = t_df[
            np.isfinite(t_df["distance_pos"]) &
            np.isfinite(t_df["complexity"])
        ].copy()

        X_t      = t_df["complexity"].values.astype(np.float64).reshape(-1, 1)
        expected = np.clip(model.predict(X_t), 1e-6, None)
        t_df = t_df.copy()
        t_df["norm_score"] = t_df["distance_pos"].values.astype(np.float64) / expected

        ax.hist(t_df[t_df["is_real"]]["norm_score"],
                bins=30, alpha=0.6, color="steelblue", label="Real", density=True)
        ax.hist(t_df[~t_df["is_real"]]["norm_score"],
                bins=30, alpha=0.6, color="tomato", label="Generated", density=True)
        ax.axvline(x=1.0, color="black", linestyle="--", lw=1.5, label="Threshold=1.0")
        ax.set_xlabel("Normalized Score")
        ax.set_ylabel("Density")
        ax.set_title(f"Score Distribution\n"
                     f"({transform_labels.get(transform, transform)})")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = args.output_dir / "robustness_results.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_png}")
    plt.show()

    # ----------------------------------------------------------------
    # 6. Summary table
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Transform':<12} {'AP_orig':>10} {'AP_norm':>10} "
          f"{'Improvement':>12} {'d_orig':>8} {'d_norm':>8}")
    print("-"*60)
    for _, row in results_df.iterrows():
        print(f"{row['transform']:<12} {row['ap_original']:>10.4f} "
              f"{row['ap_normalized']:>10.4f} {row['improvement']:>+12.4f} "
              f"{row['cohen_d_orig']:>8.2f} {row['cohen_d_norm']:>8.2f}")
    print(f"{'='*60}")

    merged_out = dist_df.merge(complexity_df, on=["file", "dir"], how="inner")
    merged_out.to_csv(args.output_dir / "full_results.csv", index=False)
    print("\nDone!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distances-parquet", type=Path,
                        default=Path("output/01/jpeg_all/distances.parquet"))
    parser.add_argument("--real-dir",  type=Path,
                        default=Path("data/raw/real"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("output/04/jpeg_robustness_all"))
    parser.add_argument("--distance-metric", default="lpips_vgg_2")
    parser.add_argument("--amount", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())