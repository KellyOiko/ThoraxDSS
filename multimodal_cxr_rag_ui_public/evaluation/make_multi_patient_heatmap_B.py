"""
Generate Block B heatmaps for multi-patient sweep experiments.

This script loads per-patient sweep outputs, builds patient-by-setting matrices
for retrieved-case faithfulness metrics, and saves a two-panel heatmap figure.
"""
import argparse
import json
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

WEIGHTS_NOTE = "T2T=text→text, I2I=image→image, T2I=text→image"


def truncated_cmap(name: str, minval: float = 0.0, maxval: float = 1.0, n: int = 256):
    """Return a truncated version of a matplotlib colormap."""
    base = cm.get_cmap(name, n)
    new_colors = base(np.linspace(minval, maxval, n))
    return colors.LinearSegmentedColormap.from_list(f"{name}_{minval}_{maxval}", new_colors)

METRICS = [
    ("mean_case_rougeL",
     "Mean ROUGE-L (retrieved case findings vs case reports)",
     truncated_cmap("cividis", 0.05, 0.85)),   # Softer cividis range, avoiding very bright yellow.

    ("mean_case_bertscore",
     "Mean BERTScore F1 (retrieved case findings vs case reports)",
     truncated_cmap("viridis", 0.05, 0.70)),   # Viridis range restricted to blue–green tones, avoiding bright yellow.
]


PATIENT_RE = re.compile(r"patient_(\d+)\.json$")

PREFERRED_WEIGHT_ORDER = [
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.5, 0.25, 0.25),
    (0.25, 0.5, 0.25),
    (0.25, 0.25, 0.5),
]
ORDER_INDEX = {w: i for i, w in enumerate(PREFERRED_WEIGHT_ORDER)}



def _find_latest_n_folder(root: Path) -> Path:
    """
    Picks the most recently modified folder under evaluation/_multi_patient_sweep/
    that contains patient_*.json.
    """
    candidates = []
    for p in root.glob("*"):
        if p.is_dir() and any(p.glob("patient_*.json")):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No folder with patient_*.json found under: {root}")
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_patient_file(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path.name} must be a list of per-setting entries")
    return data


def _short_label_from_weights(w: dict) -> str:
    # Build a compact weight label such as (1,0,0) or (0.25,0.5,0.25).
    def fmt(v: float) -> str:
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.2f}".rstrip("0").rstrip(".")
    t2t = float(w.get("t2t", 0.0) or 0.0)
    i2i = float(w.get("i2i", 0.0) or 0.0)
    t2i = float(w.get("t2i", 0.0) or 0.0)
    return f"({fmt(t2t)},{fmt(i2i)},{fmt(t2i)})"


def _weights_tuple(w: dict):
    # Return a stable tuple representation of the setting for ordering and lookup.
    t2t = float(w.get("t2t", 0.0) or 0.0)
    i2i = float(w.get("i2i", 0.0) or 0.0)
    t2i = float(w.get("t2i", 0.0) or 0.0)
    return (t2t, i2i, t2i)


def build_matrices(exp_dir: Path):
    """Build per-metric patient-by-setting matrices from multi-patient sweep JSON files."""

    patient_files = sorted(
        exp_dir.glob("patient_*.json"),
        key=lambda p: int(PATIENT_RE.search(p.name).group(1)) if PATIENT_RE.search(p.name) else 10**9
    )
    if not patient_files:
        raise FileNotFoundError(f"No patient_*.json in {exp_dir}")

    # First pass: collect all weight settings so the heatmap columns stay consistent across patients.
    settings = {}  # key=weights_tuple -> label
    for pf in patient_files:
        entries = _load_patient_file(pf)
        for e in entries:
            w = (e.get("weights") or {})
            key = _weights_tuple(w)
            settings[key] = _short_label_from_weights(w)


    sorted_keys = sorted(settings.keys(), key=lambda w: ORDER_INDEX.get(tuple(round(x, 2) for x in w), 999))

    # Sort columns using the preferred weight-order configuration.
    col_labels = [settings[k] for k in sorted_keys]

    # One heatmap matrix per metric, with shape [n_patients, n_settings].
    n_pat = len(patient_files)
    n_set = len(sorted_keys)
    mats = {mkey: np.full((n_pat, n_set), np.nan, dtype=float) for mkey, *_ in METRICS}

    row_labels = []
    for i, pf in enumerate(patient_files):
        m = PATIENT_RE.search(pf.name)
        pid = int(m.group(1)) if m else (i + 1)
        row_labels.append(f"P{pid:03d}")

        entries = _load_patient_file(pf)
        
        # Map each weight setting to its stored result for faster lookup.
        by_key = {}
        for e in entries:
            w = (e.get("weights") or {})
            key = _weights_tuple(w)
            by_key[key] = e

        for j, key in enumerate(sorted_keys):
            e = by_key.get(key)
            if not e:
                continue
            s = (e.get("summary") or {})
            for metric_key, *_rest in METRICS:
                v = s.get(metric_key)
                if isinstance(v, (int, float)):
                    mats[metric_key][i, j] = float(v)

    return row_labels, col_labels, mats


def _draw_heatmap(ax, data, title, cmap, vmin=0.0, vmax=1.0, annotate=True):
    """Draw a single annotated heatmap on the provided matplotlib axis."""
    
    im = ax.imshow(data, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(-.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5, alpha=0.35)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.set_title(title)
    ax.set_xlabel("Fusion weights (T2T, I2I, T2I)")
    ax.set_ylabel("Patient")

    # Tick positions and labels are assigned by the caller.
    if annotate:
        nrows, ncols = data.shape
        for r in range(nrows):
            for c in range(ncols):
                val = data[r, c]
                if np.isfinite(val):
                    # Choose annotation color based on the background intensity for readability.
                    text_color = "white" if val < 0.45 else "black"
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=8, color=text_color)

                   
    return im


def main():
    """Load the latest multi-patient sweep folder and generate the Block B heatmap figure."""

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp",
        type=str,
        default="",
        help="Path to experiment folder (e.g., evaluation/_multi_patient_sweep/n10). "
             "If omitted, picks latest under evaluation/_multi_patient_sweep/."
    )
    args = ap.parse_args()

    root = Path("evaluation/_multi_patient_sweep").resolve()
    exp_dir = Path(args.exp).resolve() if args.exp else _find_latest_n_folder(root)

    row_labels, col_labels, mats = build_matrices(exp_dir)

    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Create a two-panel figure with one heatmap per metric.
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 8), constrained_layout=True)

    ims = []
    for ax, (metric_key, metric_title, cmap) in zip(axes, METRICS):
        im = _draw_heatmap(
            ax=ax,
            data=mats[metric_key],
            title=metric_title,
            cmap=cmap,
            vmin=0.0, vmax=1.0,
            annotate=True,
        )
        ims.append(im)

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=20, ha="right")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)

        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Score (0–1)")

    # Add a small note explaining the retrieval-weight abbreviations.
    fig.text(0.01, 0.01, WEIGHTS_NOTE, fontsize=10)

    out_path = fig_dir / "B_case_faithfulness_heatmap.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print("\n✅ Saved heatmap figure to:")
    print(" -", str(out_path))
    print("Used experiment folder:")
    print(" -", str(exp_dir))


if __name__ == "__main__":
    main()
