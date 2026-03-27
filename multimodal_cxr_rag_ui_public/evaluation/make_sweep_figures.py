# evaluation/make_sweep_figures.py
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


# Metrics to visualize: (metric_key, plot_title, y_label, clamp_to_0_1)
METRICS_TO_PLOT = [
    ("patient_rougeL",     "Patient ROUGE-L (patient_findings vs patient report)", "Score (0–1)", True),
    ("patient_bertscore",  "Patient BERTScore F1 (patient_findings vs patient report)", "Score (0–1)", True),
    ("patient_chrf",       "Patient chrF (patient_findings vs patient report)", "chrF", False),
    ("patient_bleu",       "Patient BLEU (patient_findings vs patient report)", "BLEU", False),

    ("mean_case_rougeL",   "Mean ROUGE-L (retrieved case findings vs reports)", "Score (0–1)", True),
    ("mean_case_bertscore","Mean BERTScore F1 (retrieved case findings vs reports)", "Score (0–1)", True),
    ("mean_common_f1",     "Mean F1: common_with_patient (term-based)", "F1 (0–1)", True),
    ("mean_diff_f1",       "Mean F1: differences_from_patient (term-based)", "F1 (0–1)", True),

    ("final_rougeL",       "Final ROUGE-L (final diagnosis vs patient report)", "Score (0–1)", True),
    ("final_bertscore",    "Final BERTScore F1 (final diagnosis vs patient report)", "Score (0–1)", True),

    ("similarity_accuracy","Similarity label accuracy (weak gold)", "Accuracy (0–1)", True),
    ("similarity_macro_f1","Similarity macro-F1 (weak gold)", "F1 (0–1)", True),
  
    ("n_warnings",         "Validation warnings", "Count", False),
]

# Short note shown in figures to explain the weight abbreviations.
WEIGHTS_NOTE = "T2T=text→text, T2I=text→image, I2I=image→image"

# Preferred display order for weight settings in plots and tables.
PREFERRED_WEIGHT_ORDER = [
    (1.0, 0.0, 0.0),   # T2T-only
    (0.0, 0.0, 1.0),   # I2I-only
    (0.0, 1.0, 0.0),   # T2I-only
    (0.5, 0.25, 0.25), # Mixed (T2T-heavy)
    (0.25, 0.25, 0.5), # Mixed (I2I-heavy)
    (0.25, 0.5, 0.25), # Mixed (T2I-heavy)
]
ORDER_INDEX = {w: i for i, w in enumerate(PREFERRED_WEIGHT_ORDER)}

def _is_multi_patient_json(obj) -> bool:
    """Detect whether the loaded JSON follows the aggregated multi-patient sweep format."""
    return isinstance(obj, dict) and any(isinstance(k, str) and "T2T=" in k and "I2I=" in k and "T2I=" in k for k in obj.keys())

def _parse_multi_key(key: str):
    """
    Parse a setting key such as "T2T=0.50 I2I=0.25 T2I=0.25".
    Returns weights in thesis order: (w_t2t, w_t2i, w_i2i).
    """
    parts = key.replace(",", " ").split()
    kv = {}
    for p in parts:
        if "=" in p:
            a, b = p.split("=", 1)
            kv[a.strip()] = float(b.strip())
    w_t2t = float(kv.get("T2T", 0.0))
    w_i2i = float(kv.get("I2I", 0.0))
    w_t2i = float(kv.get("T2I", 0.0))
    # thesis order: (t2t, t2i, i2i)
    return (w_t2t, w_t2i, w_i2i)

def _extract_rows_multi_patient(data: dict):
    """
    Convert aggregated multi-patient JSON into the same row structure used for plotting.

    Expected input format:
    data[setting_key]["mean_metrics"][metric_key]
    """
    rows = []
    for setting_key, blob in data.items():
        if not isinstance(blob, dict):
            continue
        mm = blob.get("mean_metrics", {}) or {}
        if not isinstance(mm, dict):
            continue

        w_t2t, w_t2i, w_i2i = _parse_multi_key(setting_key)

        row = {
            "name": setting_key,
            "w_t2t": float(w_t2t),
            "w_t2i": float(w_t2i),
            "w_i2i": float(w_i2i),
            "label_short": _short_label(w_t2t, w_t2i, w_i2i, compact=False),
            "label_compact": _short_label(w_t2t, w_t2i, w_i2i, compact=True),
            "n_warnings": int(mm.get("n_warnings", 0)) if isinstance(mm.get("n_warnings", 0), (int, float)) else 0,
        }

        for k, _title, _ylab, _clamp in METRICS_TO_PLOT:
            if k in mm:
                row[k] = mm.get(k)

        rows.append(row)

    # Sort rows using the preferred thesis weight order.
    def _key(r):
        w = (round(r["w_t2t"], 2), round(r["w_t2i"], 2), round(r["w_i2i"], 2))
        return (ORDER_INDEX.get(w, 999), r["name"])

    rows.sort(key=_key)
    return rows



def _paired_2subplot_figure(
    x_labels,
    m1_key, m1_title, m1_ylabel, m1_clamp01,
    m2_key, m2_title, m2_ylabel, m2_clamp01,
    rows,
    out_path: Path,
    bar_color="#6B6B6B",
    edge_color="#1A1A1A",
):
    """Create a two-panel bar figure for a pair of related metrics."""
    
    x = list(range(len(x_labels)))

    v1 = [float(r.get(m1_key, 0.0) or 0.0) for r in rows]
    v2 = [float(r.get(m2_key, 0.0) or 0.0) for r in rows]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10.5, 7.2), sharex=True)

    # Top subplot
    axes[0].bar(x, v1, color=bar_color, edgecolor=edge_color, linewidth=0.6)
    axes[0].set_title(m1_title)
    axes[0].set_ylabel(m1_ylabel)
    if m1_clamp01:
        axes[0].set_ylim(0.0, 1.0)

    # Bottom subplot
    axes[1].bar(x, v2, color=bar_color, edgecolor=edge_color, linewidth=0.6)
    axes[1].set_title(m2_title)
    axes[1].set_ylabel(m2_ylabel)
    if m2_clamp01:
        axes[1].set_ylim(0.0, 1.0)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels, rotation=20, ha="right")
    axes[1].set_xlabel(r"Fusion weights ($w_{t2t}$, $w_{t2i}$, $w_{i2i}$)")

    fig.text(0.01, 0.01, WEIGHTS_NOTE, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _find_latest_experiment_folder(root: Path) -> Path:
    """Return the most recent sweep experiment folder containing sweep_summary.json."""

    candidates = []
    for p in root.glob("*"):
        if p.is_dir() and (p / "sweep_summary.json").exists():
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No experiment folders with sweep_summary.json found under: {root}")
    candidates.sort(key=lambda x: (x / "sweep_summary.json").stat().st_mtime, reverse=True)
    return candidates[0]


def _load_sweep(path: Path):
    """Load a single-case sweep summary JSON file and validate its top-level structure."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("sweep_summary.json must be a list of entries")
    return data


def _fmt_w(x: float) -> str:
    # Format weights consistently for plot labels.
    return f"{x:.2f}"


def _short_label(w_t2t: float, w_t2i: float, w_i2i: float, compact: bool = False) -> str:
    """
    Return an x-axis label in thesis order: (T2T, T2I, I2I).
    - compact=False -> "(1.00, 0.00, 0.00)"
    - compact=True  -> "(1,0,0)" if very close to ints
    """
    if compact:
        def as_intish(v: float) -> str:
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            return _fmt_w(v)
        return f"({as_intish(w_t2t)},{as_intish(w_t2i)},{as_intish(w_i2i)})"
    return f"({_fmt_w(w_t2t)}, {_fmt_w(w_t2i)}, {_fmt_w(w_i2i)})"


def _extract_rows(entries):
    """
    Extract plotting rows from either supported sweep-summary schema.

    Supported formats:
    A) nested: {"name", "weights", "summary", "validation"}
    B) flat: metrics at top level plus "_weights_effective"/"_weights_requested"
    """
    rows = []
    for e in entries:
        if not isinstance(e, dict):
            continue

        name = e.get("name", "unknown")

        summary = e.get("summary")
        if not isinstance(summary, dict):
            summary = e  # flat schema

        weights = e.get("weights")
        if not isinstance(weights, dict):
            weights = e.get("_weights_effective")
        if not isinstance(weights, dict):
            weights = {}

        validation = e.get("validation")
        if not isinstance(validation, dict):
            validation = {}

        w_t2t = float(weights.get("t2t", 0.0) or 0.0)
        w_i2i = float(weights.get("i2i", 0.0) or 0.0)
        w_t2i = float(weights.get("t2i", 0.0) or 0.0)

        row = {
            "name": name,
            "w_t2t": w_t2t,
            "w_i2i": w_i2i,
            "w_t2i": w_t2i,
            "label_short": _short_label(w_t2t, w_t2i, w_i2i, compact=False),
            "label_compact": _short_label(w_t2t, w_t2i, w_i2i, compact=True),

            "n_warnings": (
                int(summary.get("n_warnings"))
                if isinstance(summary.get("n_warnings"), (int, float))
                else len(validation.get("warnings", []) or [])
            ),
        }

        for k, _title, _ylab, _clamp in METRICS_TO_PLOT:
            if k in summary:
                row[k] = summary.get(k)

        rows.append(row)

    def _key(r):
        w = (round(r["w_t2t"], 2), round(r["w_t2i"], 2), round(r["w_i2i"], 2))

        return (ORDER_INDEX.get(w, 999), r["name"])

    # Sort rows in a stable order by preferred weights, then by name.
    rows.sort(key=_key)
    
    return rows


def _save_table_csv(rows, out_path: Path):
    """Save the extracted sweep rows as a CSV table for reporting and plotting."""

    cols = ["name", "w_t2t", "w_t2i", "w_i2i"] + [k for k, *_rest in METRICS_TO_PLOT]
    lines = [",".join(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, "")
            s = "" if v is None else str(v)
            if "," in s or '"' in s:
                s = '"' + s.replace('"', '""') + '"'
            vals.append(s)
        lines.append(",".join(vals))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _bar_plot(
    x_labels,
    values,
    title,
    ylabel,
    out_path: Path,
    clamp01: bool,
    bar_color: str = "#5A7D9A",  # muted steel-ish (not bright blue)
):
    """Create and save a single-metric bar plot across retrieval-weight settings."""

    plt.figure(figsize=(10, 4.8))
    plt.bar(range(len(x_labels)), values, color=bar_color)

    plt.xticks(range(len(x_labels)), x_labels, rotation=20, ha="right")

    plt.xlabel(r"Fusion weights ($w_{t2t}$, $w_{t2i}$, $w_{i2i}$)")

    plt.ylabel(ylabel)
    plt.title(title)

    # small explanation note inside plot area (for the weight abbreviations)
    plt.gcf().text(0.01, 0.01, WEIGHTS_NOTE, fontsize=9)

    if clamp01:
        plt.ylim(0.0, 1.0)

    plt.tight_layout(rect=[0, 0, 1, 0.93])


    plt.savefig(out_path, dpi=220, bbox_inches="tight")

    plt.close()

def _weights_stacked_plot(rows, out_path: Path):
    """Create a stacked bar plot showing the fusion weights for each evaluated setting."""

    names = [r["label_compact"] for r in rows]
    w_t2t = [r["w_t2t"] for r in rows]
    w_t2i = [r["w_t2i"] for r in rows]
    w_i2i = [r["w_i2i"] for r in rows]

    # Font sizes tuned for thesis-ready figures.
    FS_TITLE = 18
    FS_AXIS  = 14
    FS_TICK  = 12
    FS_NOTE  = 13
    FS_LEG   = 14

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = range(len(names))

    # Okabe–Ito (color-blind safe)
    c_t2t = "#0072B2"  # blue
    c_t2i = "#009E73"  # green
    c_i2i = "#E69F00"  # orange

    edge = "#222222"
    lw = 0.6

    b1 = ax.bar(
        x, w_t2t, label="T2T", color=c_t2t,
        edgecolor=edge, linewidth=lw, hatch="///"
    )
    b2 = ax.bar(
        x, w_t2i, bottom=w_t2t, label="T2I", color=c_t2i,
        edgecolor=edge, linewidth=lw, hatch="xxx"
    )
    bottom2 = [w_t2t[i] + w_t2i[i] for i in range(len(names))]
    b3 = ax.bar(
        x, w_i2i, bottom=bottom2, label="I2I", color=c_i2i,
        edgecolor=edge, linewidth=lw, hatch="..."
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=FS_TICK)

    ax.set_xlabel(r"Fusion weights $(w_{t2t},\, w_{t2i},\, w_{i2i})$", fontsize=FS_AXIS)

    ax.set_ylabel("Weight (sum=1)", fontsize=FS_AXIS)
    ax.tick_params(axis="y", labelsize=FS_TICK)
    ax.set_ylim(0, 1.0)

    # Keep the title close to the plot so it does not collide with the legend.
    ax.set_title(
        "Retrieval fusion weights per setting (stacked)",
        fontsize=FS_TITLE,
        pad=2,    # Keep the title close to the axes to avoid legend overlap.
    )

    # Place the legend above the plot while preserving the T2T–T2I–I2I order.
    ax.legend(
        handles=[b1, b2, b3],
        labels=["T2T", "T2I", "I2I"],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),   # Position the legend above the title area.
        ncols=3,
        frameon=False,
        fontsize=FS_LEG,
        borderaxespad=0.0
    )

    # Place the explanatory note in figure coordinates for stable positioning.
    fig.text(0.01, 0.02, WEIGHTS_NOTE, fontsize=FS_NOTE)

    # Use fixed margins to keep the thesis layout stable across runs.
    fig.subplots_adjust(top=0.83, bottom=0.30)

    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    """Load a sweep summary, generate per-metric figures, and save thesis-ready plot outputs."""

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp",
        type=str,
        default="",
        help="Path to experiment folder (the one containing sweep_summary.json). "
             "If omitted, picks the latest under evaluation/_sweeps/."
    )
    ap.add_argument(
        "--compact_xticks",
        action="store_true",
        help='Use compact x tick labels like "(1,0,0)" instead of "(1.00, 0.00, 0.00)".'
    )
    ap.add_argument("--json", type=str, default="", help="Optional path to JSON file...")


    args = ap.parse_args()

    sweeps_root = Path("evaluation/_sweeps").resolve()
    exp_dir = Path(args.exp).resolve() if args.exp else _find_latest_experiment_folder(sweeps_root)

    
    if args.json:
        sweep_path = Path(args.json).resolve()
    else:
        sweep_path = exp_dir / "sweep_summary.json"

    data = json.loads(sweep_path.read_text(encoding="utf-8"))

    if _is_multi_patient_json(data):
        rows = _extract_rows_multi_patient(data)
    else:
        if not isinstance(data, list):
            raise ValueError("sweep_summary.json must be a list (single-case) OR a dict (multi-patient).")
        rows = _extract_rows(data)

    if not rows:
        raise RuntimeError("No rows extracted from sweep_summary.json")

    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _save_table_csv(rows, fig_dir / "sweep_table_for_thesis.csv")

    x_labels = [r["label_compact"] if args.compact_xticks else r["label_short"] for r in rows]

    # Create one bar plot for each metric in the sweep summary.
    for metric_key, metric_title, ylab, clamp01 in METRICS_TO_PLOT:
        values = []
        for r in rows:
            v = r.get(metric_key, None)
            values.append(float(v) if isinstance(v, (int, float)) else 0.0)

        out_png = fig_dir / f"{metric_key}.png"
        _bar_plot(
            x_labels=x_labels,
            values=values,
            title=metric_title,
            ylabel=ylab,
            out_path=out_png,
            clamp01=clamp01,
        )

    # Create paired thesis figures for similarity and comparison-quality metrics.
    _paired_2subplot_figure(
        x_labels=x_labels,
        m1_key="similarity_accuracy",
        m1_title="Similarity label accuracy (weak gold)",
        m1_ylabel="Accuracy (0–1)",
        m1_clamp01=True,
        m2_key="similarity_macro_f1",
        m2_title="Similarity macro-F1 (weak gold)",
        m2_ylabel="F1 (0–1)",
        m2_clamp01=True,
        rows=rows,
        out_path=fig_dir / "C_similarity_labels.png",
    )

    _paired_2subplot_figure(
        x_labels=x_labels,
        m1_key="mean_common_f1",
        m1_title="Mean F1: common_with_patient (term-based)",
        m1_ylabel="F1 (0–1)",
        m1_clamp01=True,
        m2_key="mean_diff_f1",
        m2_title="Mean F1: differences_from_patient (term-based)",
        m2_ylabel="F1 (0–1)",
        m2_clamp01=True,
        rows=rows,
        out_path=fig_dir / "D_comparison_quality.png",
    )


    # Create a stacked plot showing the fusion-weight composition of each setting.
    _weights_stacked_plot(rows, fig_dir / "weights_stacked.png")

    print("\n✅ Figures saved to:")
    print(" -", str(fig_dir))
    print("Includes: one png per metric + weights_stacked.png + sweep_table_for_thesis.csv")
    print("\nUsed experiment folder:")
    print(" -", str(exp_dir))


if __name__ == "__main__":
    main()
