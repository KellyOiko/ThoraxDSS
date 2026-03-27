"""
Generate thesis-style heatmaps for multi-patient retrieval-weight sweep results.
"""
from __future__ import annotations

from pathlib import Path
import sys
import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap

# Force an ASCII-only project root because some Windows/FAISS setups are unreliable with Unicode paths.
PREFERRED_ROOT = Path(r"C:\mm_cxr")
if not (PREFERRED_ROOT / "backend" / "main.py").exists():
    raise RuntimeError(
        f"Expected ASCII clone at {PREFERRED_ROOT} but backend/main.py not found.\n"
        f"Either clone/copy your project to C:\\mm_cxr, or change PREFERRED_ROOT."
    )

PROJECT_ROOT = PREFERRED_ROOT
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# Default input/output locations for the heatmap generation script.
DEFAULT_JSON = PROJECT_ROOT / "evaluation" / "_multi_patient_sweep" / "n10" / "multi_patient_weight_sweep_summary_v2.json"
DEFAULT_OUTDIR = PROJECT_ROOT / "evaluation" / "_multi_patient_sweep" / "n10" / "Figures"

WEIGHTS_NOTE = "T2T=text→text, T2I=text→image, I2I=image→image"


# Metrics for block B: retrieved-case text quality.
BLOCK_B_METRICS = [
    ("mean_case_bleu", "mean_case_bleu"),
    ("mean_case_rougeL", "mean_case_rougeL"),
    ("mean_case_meteor", "mean_case_meteor"),
    ("mean_case_chrf", "mean_case_chrf"),
    ("mean_case_bertscore", "mean_case_bertscore"),
]

# Metrics for blocks C and D: similarity-label and comparison-summary quality.
BLOCK_CD_METRICS = [
    ("similarity_accuracy", "similarity_accuracy"),
    ("similarity_macro_f1", "similarity_macro_f1"),
    ("mean_common_f1", "mean_common_f1"),
    ("mean_diff_f1", "mean_diff_f1"),
]

# Metrics for block E: final diagnosis assessment quality.
BLOCK_E_METRICS = [
    ("final_bleu", "final_bleu"),
    ("final_rougeL", "final_rougeL"),
    ("final_meteor", "final_meteor"),
    ("final_chrf", "final_chrf"),
    ("final_bertscore", "final_bertscore"),
]

# Fixed display order for retrieval-weight settings in the final figures.
SETTING_ORDER = [
    ("T2T=1.00 T2I=0.00 I2I=0.00", "T2T-only", (1.00, 0.00, 0.00)),
    ("T2T=0.00 T2I=0.00 I2I=1.00", "I2I-only", (0.00, 0.00, 1.00)),
    ("T2T=0.00 T2I=1.00 I2I=0.00", "T2I-only", (0.00, 1.00, 0.00)),
    ("T2T=0.50 T2I=0.25 I2I=0.25", "Mixed (T2T-heavy)", (0.50, 0.25, 0.25)),
    ("T2T=0.25 T2I=0.25 I2I=0.50", "Mixed (I2I-heavy)", (0.25, 0.25, 0.50)),
    ("T2T=0.25 T2I=0.50 I2I=0.25", "Mixed (T2I-heavy)", (0.25, 0.50, 0.25)),
]



# Strong blue-to-yellow colormap for the heatmaps.
CMAP = LinearSegmentedColormap.from_list(
    "strong_blue_to_yellow",
    ["#1f4e99", "#4ea3d8", "#fff200"]  # low -> medium -> high
)

def load_json(path: Path) -> Dict:
    """Load the summary JSON file from disk."""

    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def extract_matrix(data: Dict, metrics: List[Tuple[str, str]], normalize: bool) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Build the raw and plotting matrices for one selected metric block.

    Returns:
    - raw metric values,
    - plotting matrix (raw or per-metric normalized),
    - row labels,
    - column labels.
    """
    
    row_labels: List[str] = []
    raw_rows: List[List[float]] = []

    for json_key, short_label, w in SETTING_ORDER:
        if json_key not in data:
            raise KeyError(f"Missing key in JSON: '{json_key}'")

        mm = data[json_key].get("mean_metrics", {}) or {}
        row: List[float] = []
        for metric_key, _pretty in metrics:
            if metric_key not in mm:
                raise KeyError(f"Missing metric '{metric_key}' under mean_metrics for '{json_key}'")
            row.append(float(mm[metric_key]))

        
        row_labels.append(f"{short_label}  ({w[0]:.2f},{w[1]:.2f},{w[2]:.2f})")

        raw_rows.append(row)

    # Convert the collected rows into a numeric matrix in the fixed setting order.
    raw = np.array(raw_rows, dtype=float)

    # Optionally normalize each metric column independently for visual comparison.
    if not normalize:
        plot_mat = raw.copy()
    else:
        plot_mat = np.zeros_like(raw)
        for j in range(raw.shape[1]):
            col = raw[:, j]
            cmin, cmax = float(np.min(col)), float(np.max(col))
            plot_mat[:, j] = 0.0 if np.isclose(cmax, cmin) else (col - cmin) / (cmax - cmin)

    col_labels = [pretty for _k, pretty in metrics]
    return raw, plot_mat, row_labels, col_labels

def plot_heatmap(raw: np.ndarray,
                 mat: np.ndarray,
                 row_labels: List[str],
                 col_labels: List[str],
                 normalize: bool,
                 title: str,
                 subtitle: str,
                 out_pdf: Path,
                 out_png: Path,
                 highlight_row_label: str | None = None) -> None:
    """
    Render and save one heatmap figure.

    The plotted colors may use per-metric normalization, but the cell annotations
    always show the original raw metric values.
    """

    # Scale figure size dynamically based on the number of rows and columns.
    n_rows, n_cols = mat.shape
    fig_w = max(7.6, 1.25 * n_cols + 3.0)
    fig_h = max(4.8, 0.80 * n_rows + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Use a fixed 0–1 color scale for normalized plots; otherwise use raw-value scaling.
    if normalize:
        im = ax.imshow(mat, aspect="auto", cmap=CMAP, vmin=0.0, vmax=1.0)
    else:
        im = ax.imshow(mat, aspect="auto", cmap=CMAP)

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels(col_labels, rotation=25, ha="right")
    ax.set_yticklabels(row_labels)

    ax.set_title(title + "\n" + subtitle, pad=14)
    fig.text(0.01, 0.01, WEIGHTS_NOTE, fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized (per-metric)" if normalize else "Raw value")

    # Annotate each cell with its raw metric value.
    for i in range(n_rows):
        for j in range(n_cols):
            if normalize:
                txt_color = "black" if mat[i, j] > 0.62 else "white"
            else:
                # For raw-value plots, choose text color from the relative column intensity.
                txt_color = "black" if (mat[i, j] > (np.min(mat[:, j]) + 0.65*(np.max(mat[:, j]) - np.min(mat[:, j])))) else "white"
            ax.text(j, i, f"{raw[i, j]:.4f}", ha="center", va="center", fontsize=10, color=txt_color)

    # Highlight the best setting for each metric using the raw values.
    best_rows = np.argmax(raw, axis=0)
    for j in range(n_cols):
        i_best = int(best_rows[j])
        rect = Rectangle((j - 0.5, i_best - 0.5), 1, 1,
                         fill=False, linewidth=3.2, edgecolor="black")
        ax.add_patch(rect)

    # Optionally outline one selected row, e.g. T2T-only.
    if highlight_row_label and (highlight_row_label in row_labels):
        irow = row_labels.index(highlight_row_label)
        rect_row = Rectangle((-0.5, irow - 0.5), n_cols, 1,
                             fill=False, linewidth=2.0, edgecolor="black")
        ax.add_patch(rect_row)

    plt.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved: {out_pdf}")
    print(f"✅ Saved: {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
   
    ap.add_argument("--block", choices=["B", "CD", "E"], default="B",
        help="Which block to plot: B (mean_case), CD (agreement), E (final metrics).")

   
   
    ap.add_argument("--no-normalize", action="store_true",
                    help="Disable per-metric normalization (recommended for CD since all are 0-1).")
    ap.add_argument("--highlight", default="T2T-only",
                    help="Row label to outline (optional).")
    args = ap.parse_args()

    # Resolve input/output paths against the forced project root when needed.
    json_path = Path(args.json)
    if not json_path.is_absolute():
        json_path = (PROJECT_ROOT / json_path).resolve()

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = (PROJECT_ROOT / outdir).resolve()

    data = load_json(json_path)

    # Select the metric group, output filenames, and normalization behavior for the requested block.
    if args.block == "B":
        metrics = BLOCK_B_METRICS
        title = "Mean retrieved-case metrics (Block B) across fusion settings (n=10, k=3)"
        subtitle = "Color = per-metric min–max normalization; text = raw values" if not args.no_normalize \
                   else "Raw values (colors not comparable across metrics)"
        out_pdf = outdir / "n10_blockB_mean_case_heatmap.pdf"
        out_png = outdir / "n10_blockB_mean_case_heatmap.png"
        normalize = (not args.no_normalize)


    elif args.block == "E":
        metrics = BLOCK_E_METRICS
        title = "Final assessment metrics (Block E) across fusion settings (n=10, k=3)"
        
        # For block E, normalization is usually more informative because the metrics have different scales.
        normalize = (not args.no_normalize)
        subtitle = "Color = per-metric min–max normalization; text = raw values" if normalize \
                else "Raw values (colors not comparable across metrics)"
        out_pdf = outdir / "n10_blockE_final_metrics_heatmap.pdf"
        out_png = outdir / "n10_blockE_final_metrics_heatmap.png"

    else:  # CD
        metrics = BLOCK_CD_METRICS
        title = "Agreement metrics (Blocks C + D) across fusion settings (n=10, k=3)"
        # For block CD, raw values are already directly comparable because they lie on the same 0–1 scale.
        normalize = (not args.no_normalize)
        subtitle = "Raw values (0–1; directly comparable)" if args.no_normalize else "Color = per-metric min–max normalization; text = raw values"
        out_pdf = outdir / "n10_blockCD_agreement_heatmap.pdf"
        out_png = outdir / "n10_blockCD_agreement_heatmap.png"

    # Build the numeric matrix and plot labels for the selected block.
    raw, mat, row_labels, col_labels = extract_matrix(data, metrics=metrics, normalize=normalize)

    # Generate and save the final heatmap figure.
    plot_heatmap(
        raw=raw,
        mat=mat,
        row_labels=row_labels,
        col_labels=col_labels,
        normalize=normalize,
        title=title,
        subtitle=subtitle,
        out_pdf=out_pdf,
        out_png=out_png,
        highlight_row_label=args.highlight,
    )

if __name__ == "__main__":
    main()
