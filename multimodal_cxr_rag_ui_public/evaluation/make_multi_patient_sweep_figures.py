"""
Generate aggregate figures for multi-patient retrieval-weight sweep experiments.
"""
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt


WEIGHTS_NOTE = "T2T=text→text, T2I=text→image, I2I=image→image"

# Keep a fixed thesis-friendly order for retrieval-weight settings.
SETTINGS_ORDER = [
    "T2T=1.00 I2I=0.00 T2I=0.00",
    "T2T=0.00 I2I=1.00 T2I=0.00",
    "T2T=0.00 I2I=0.00 T2I=1.00",
    "T2T=0.50 I2I=0.25 T2I=0.25",
    "T2T=0.25 I2I=0.50 T2I=0.25",
    "T2T=0.25 I2I=0.25 T2I=0.50",
]

# Metric blocks A–E, each visualized with two key metrics.
BLOCKS = [
    ("A_patient_faithfulness",
     ("patient_rougeL", "Patient ROUGE-L (patient_findings vs patient report)", "Score (0–1)", True),
     ("patient_bertscore", "Patient BERTScore F1 (patient_findings vs patient report)", "Score (0–1)", True)),

    ("B_retrieved_case_faithfulness",
     ("mean_case_rougeL", "Mean ROUGE-L (retrieved case findings vs case reports)", "Score (0–1)", True),
     ("mean_case_bertscore", "Mean BERTScore F1 (retrieved case findings vs case reports)", "Score (0–1)", True)),

    ("C_similarity_labels",
     ("similarity_accuracy", "Similarity label accuracy (weak gold)", "Accuracy (0–1)", True),
     ("similarity_macro_f1", "Similarity macro-F1 (weak gold)", "F1 (0–1)", True)),

    ("D_comparison_quality",
     ("mean_common_f1", "Mean F1: common_with_patient (term-based)", "F1 (0–1)", True),
     ("mean_diff_f1", "Mean F1: differences_from_patient (term-based)", "F1 (0–1)", True)),

    ("E_final_assessment",
     ("final_rougeL", "Final ROUGE-L (final diagnosis vs patient report)", "Score (0–1)", True),
     ("final_bertscore", "Final BERTScore F1 (final diagnosis vs patient report)", "Score (0–1)", True)),
]

# Neutral bar styling for thesis-ready plots.
BAR_COLOR = "#6B6B6B"  # professional neutral gray
EDGE_COLOR = "#1A1A1A"


def _mean_std(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Return the mean and sample standard deviation of a numeric list."""
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    mu = sum(vals) / len(vals)
    var = sum((x - mu) ** 2 for x in vals) / (len(vals) - 1)  # sample variance
    return mu, math.sqrt(var)


def _fmt_w(v: float) -> str:
    # Format weights compactly for axis labels, e.g. 1.00 -> "1".
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    # keep 2 decimals but strip trailing zeros
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s


def _compact_weight_label(weights: Dict) -> str:
    """Build a compact weight label in thesis order: (T2T, T2I, I2I)."""
    t2t = float(weights.get("t2t", 0.0) or 0.0)
    t2i = float(weights.get("t2i", 0.0) or 0.0)
    i2i = float(weights.get("i2i", 0.0) or 0.0)
    return f"({_fmt_w(t2t)},{_fmt_w(t2i)},{_fmt_w(i2i)})"


def _find_latest_n_folder(root: Path) -> Path:
    """
    Return the most recently modified experiment folder under root
    that contains patient_*.json files.
    """
    candidates = []
    for p in root.glob("*"):
        if p.is_dir() and list(p.glob("patient_*.json")):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f"No folders with patient_*.json under: {root}")
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_patient_file(path: Path) -> List[dict]:
    """Load one per-patient sweep result file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path.name} must be a list")
    return data


def _collect_from_patients(exp_dir: Path):
    """
    Collect per-setting metrics, validation counts, and weight metadata
    from all patient result files in one experiment directory.
    """
        
    patient_files = sorted(exp_dir.glob("patient_*.json"))
    if not patient_files:
        raise FileNotFoundError(f"No patient_*.json found in: {exp_dir}")

    # Aggregate per-setting values across all patients.
    metrics: Dict[str, Dict[str, List[float]]] = {}
    weights_by_name: Dict[str, Dict] = {}
    warnings_by_name: Dict[str, List[int]] = {}
    ok_by_name: Dict[str, List[int]] = {}

    for pf in patient_files:
        entries = _load_patient_file(pf)
        for e in entries:
            if not isinstance(e, dict):
                continue
            name = e.get("name")
            if not isinstance(name, str) or not name:
                continue

            weights = e.get("weights") or {}
            if isinstance(weights, dict) and name not in weights_by_name:
                weights_by_name[name] = weights

            summary = e.get("summary") or {}
            if name not in metrics:
                metrics[name] = {}

            # Collect numeric summary metrics for later mean/std aggregation.
            if isinstance(summary, dict):
                for k, v in summary.items():
                    if isinstance(v, (int, float)):
                        metrics[name].setdefault(k, []).append(float(v))

            # Collect validation warning counts and ok flags.
            validation = e.get("validation") or {}
            if isinstance(validation, dict):
                warn_count = len(validation.get("warnings", []) or [])
                warnings_by_name.setdefault(name, []).append(int(warn_count))
                ok_flag = 1 if validation.get("ok") else 0
                ok_by_name.setdefault(name, []).append(int(ok_flag))

    return {
        "patient_files": patient_files,
        "metrics": metrics,
        "weights_by_name": weights_by_name,
        "warnings_by_name": warnings_by_name,
        "ok_by_name": ok_by_name,
    }


def _ordered_names(metrics: Dict[str, Dict[str, List[float]]]) -> List[str]:
    """Return setting names in the preferred thesis order, with safe fallback for extras."""

    present = set(metrics.keys())
    ordered = [n for n in SETTINGS_ORDER if n in present]
    for n in sorted(present):
        if n not in ordered:
            ordered.append(n)
    return ordered


def _bar_with_error(
    x_labels: List[str],
    means: List[float],
    stds: List[float],
    title: str,
    ylabel: str,
    out_path: Path,
    clamp01: bool = True,
):
    """Create a bar plot with mean values and standard-deviation error bars."""

    plt.figure(figsize=(10.5, 4.9))
    x = list(range(len(x_labels)))

    plt.bar(
        x, means,
        yerr=stds,
        capsize=3,
        color=BAR_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.6,
        ecolor=EDGE_COLOR,
    )

    plt.xticks(x, x_labels, rotation=20, ha="right")

    plt.xlabel(r"Fusion weights ($w_{t2t}$, $w_{t2i}$, $w_{i2i}$)")

    plt.ylabel(ylabel)
    plt.title(title)

    if clamp01:
        plt.ylim(0.0, 1.0)

    plt.gcf().text(0.01, 0.01, WEIGHTS_NOTE, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _block_2subplot_figure(
    x_labels: List[str],
    top: Tuple[str, str, str, bool],
    bottom: Tuple[str, str, str, bool],
    stats: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]],
    out_path: Path,
):
    """
    Create a two-panel figure for one evaluation block.

    stats[name][metric] = (mean, std)
    """

    m1_key, m1_title, m1_ylabel, m1_clamp = top
    m2_key, m2_title, m2_ylabel, m2_clamp = bottom

    x = list(range(len(x_labels)))

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10.5, 7.2), sharex=True)

    # Top subplot
    means1, stds1 = [], []
    for name in stats.keys():
        mu, sd = stats[name].get(m1_key, (0.0, 0.0))
        means1.append(float(mu or 0.0))
        stds1.append(float(sd or 0.0))
    axes[0].bar(
        x, means1, yerr=stds1, capsize=3,
        color=BAR_COLOR, edgecolor=EDGE_COLOR, linewidth=0.6, ecolor=EDGE_COLOR
    )
    axes[0].set_ylabel(m1_ylabel)
    axes[0].set_title(m1_title)
    if m1_clamp:
        axes[0].set_ylim(0.0, 1.0)

    # Bottom subplot
    means2, stds2 = [], []
    for name in stats.keys():
        mu, sd = stats[name].get(m2_key, (0.0, 0.0))
        means2.append(float(mu or 0.0))
        stds2.append(float(sd or 0.0))
    axes[1].bar(
        x, means2, yerr=stds2, capsize=3,
        color=BAR_COLOR, edgecolor=EDGE_COLOR, linewidth=0.6, ecolor=EDGE_COLOR
    )
    axes[1].set_ylabel(m2_ylabel)
    axes[1].set_title(m2_title)
    if m2_clamp:
        axes[1].set_ylim(0.0, 1.0)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels, rotation=20, ha="right")
    #axes[1].set_xlabel("Fusion weights (T2T, I2I, T2I)")
    axes[1].set_xlabel(r"Fusion weights ($w_{t2t}$, $w_{t2i}$, $w_{i2i}$)")
    

    fig.text(0.01, 0.01, WEIGHTS_NOTE, fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _weights_stacked_plot(
    names_in_order: List[str],
    weights_by_name: Dict[str, Dict],
    x_labels: List[str],
    out_path: Path,
):
    """Create a stacked bar plot showing the fusion-weight composition of each setting."""

    w_t2t, w_t2i, w_i2i = [], [], []
    for name in names_in_order:
        # Extract weights in the same order used for the x-axis labels.
        w = weights_by_name.get(name, {}) or {}
        w_t2t.append(float(w.get("t2t", 0.0) or 0.0))
        w_t2i.append(float(w.get("t2i", 0.0) or 0.0))
        w_i2i.append(float(w.get("i2i", 0.0) or 0.0))

    x = list(range(len(x_labels)))
    plt.figure(figsize=(10.5, 4.9))

    # Okabe–Ito colors plus hatching for color-blind safety and grayscale readability.
    c_t2t = "#0072B2"  # blue
    c_t2i = "#009E73"  # green
    c_i2i = "#E69F00"  # orange
    lw = 0.6

    # Stack order follows the thesis convention: T2T -> T2I -> I2I.
    plt.bar(x, w_t2t, label="T2T", color=c_t2t,
            edgecolor=EDGE_COLOR, linewidth=lw, hatch="///")

    plt.bar(x, w_t2i, bottom=w_t2t, label="T2I", color=c_t2i,
            edgecolor=EDGE_COLOR, linewidth=lw, hatch="xxx")

    bottom2 = [w_t2t[i] + w_t2i[i] for i in range(len(x))]
    plt.bar(x, w_i2i, bottom=bottom2, label="I2I", color=c_i2i,
            edgecolor=EDGE_COLOR, linewidth=lw, hatch="...")

    plt.xticks(x, x_labels, rotation=20, ha="right")
    plt.xlabel("Fusion weights (T2T, T2I, I2I)")
    plt.ylabel("Weight (sum=1)")
    plt.ylim(0.0, 1.0)
    plt.title("Retrieval fusion weights per setting (stacked)")

    # Place the legend above the plot to avoid overlap with the stacked bars.
    plt.legend(
        frameon=False,
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        borderaxespad=0.0,
    )

    plt.gcf().text(0.01, 0.01, WEIGHTS_NOTE, fontsize=9)
    plt.tight_layout(rect=(0, 0.02, 1, 0.95))
    plt.savefig(out_path, dpi=220)
    plt.close()


def _write_aggregate_csv(
    out_path: Path,
    names_in_order: List[str],
    x_labels: List[str],
    stats: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]],
    n_patients_by_name: Dict[str, int],
    mean_warnings: Dict[str, Tuple[Optional[float], Optional[float]]],
    ok_rate: Dict[str, Tuple[Optional[float], Optional[float]]],
):
    """Write the aggregated multi-patient statistics table to CSV."""

    # Collect all metric keys available across settings.
    metric_keys = set()
    for name in names_in_order:
        for k in stats[name].keys():
            metric_keys.add(k)
    metric_keys = sorted(metric_keys)

    # Build CSV columns: identifiers first, then metric means/stds, then validation summaries.
    cols = ["name", "label_compact", "n_patients"]
    for k in metric_keys:
        cols += [f"{k}_mean", f"{k}_std"]
    cols += ["warnings_mean", "warnings_std", "validation_ok_rate_mean", "validation_ok_rate_std"]

    lines = [",".join(cols)]
    for name, lbl in zip(names_in_order, x_labels):
        row = {
            "name": name,
            "label_compact": lbl,
            "n_patients": n_patients_by_name.get(name, 0),
        }
        for k in metric_keys:
            mu, sd = stats[name].get(k, (None, None))
            row[f"{k}_mean"] = "" if mu is None else str(mu)
            row[f"{k}_std"] = "" if sd is None else str(sd)

        wmu, wsd = mean_warnings.get(name, (None, None))
        row["warnings_mean"] = "" if wmu is None else str(wmu)
        row["warnings_std"] = "" if wsd is None else str(wsd)

        okmu, oksd = ok_rate.get(name, (None, None))
        row["validation_ok_rate_mean"] = "" if okmu is None else str(okmu)
        row["validation_ok_rate_std"] = "" if oksd is None else str(oksd)

        vals = []
        for c in cols:
            s = str(row.get(c, ""))
            if "," in s or '"' in s:
                s = '"' + s.replace('"', '""') + '"'
            vals.append(s)
        lines.append(",".join(vals))

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    """Load a multi-patient sweep experiment and generate aggregate figures and CSV output."""

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exp",
        type=str,
        default="",
        help="Experiment folder with patient_*.json (e.g. evaluation/_multi_patient_sweep/n10). "
             "If omitted, picks latest under evaluation/_multi_patient_sweep/.",
    )
    args = ap.parse_args()

    root = Path("evaluation/_multi_patient_sweep").resolve()
    exp_dir = Path(args.exp).resolve() if args.exp else _find_latest_n_folder(root)

    payload = _collect_from_patients(exp_dir)
    metrics = payload["metrics"]
    weights_by_name = payload["weights_by_name"]
    warnings_by_name = payload["warnings_by_name"]
    ok_by_name = payload["ok_by_name"]

    names_in_order = _ordered_names(metrics)
    n_patients = len(payload["patient_files"])

    # Build compact x-axis labels directly from the stored fusion weights.
    x_labels = []
    for name in names_in_order:
        w = weights_by_name.get(name, {}) or {}
        x_labels.append(_compact_weight_label(w))

    # Compute per-setting means and standard deviations for all collected metrics.
    stats: Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]] = {}
    n_patients_by_name: Dict[str, int] = {}

    for name in names_in_order:
        stats[name] = {}
        # Estimate how many patient runs contributed to this setting.
        # Prefer metric-list length; fall back to validation statistics if needed.
        
        any_len = None
        for k, vals in (metrics.get(name, {}) or {}).items():
            if vals:
                any_len = len(vals)
                break
        if any_len is None:
            any_len = len(warnings_by_name.get(name, []))
        n_patients_by_name[name] = int(any_len or 0)

        for k, vals in (metrics.get(name, {}) or {}).items():
            mu, sd = _mean_std(vals)
            stats[name][k] = (mu, sd)

    # Compute aggregate validation-warning and validation-ok statistics per setting.
    warnings_stats: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    ok_rate_stats: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for name in names_in_order:
        wmu, wsd = _mean_std([float(x) for x in (warnings_by_name.get(name, []) or [])])
        warnings_stats[name] = (wmu, wsd)
        okmu, oksd = _mean_std([float(x) for x in (ok_by_name.get(name, []) or [])])
        ok_rate_stats[name] = (okmu, oksd)

    # Create the output directory for figures and tables.
    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot the stacked fusion weights for all settings.
    _weights_stacked_plot(names_in_order, weights_by_name, x_labels, fig_dir / "weights_stacked.png")

    # 2. Create one two-panel figure for each evaluation block (A–E).
    for block_name, top_metric, bottom_metric in BLOCKS:
        out_png = fig_dir / f"{block_name}.png"
        _block_2subplot_figure(
            x_labels=x_labels,
            top=top_metric,
            bottom=bottom_metric,
            stats={name: stats[name] for name in names_in_order},
            out_path=out_png,
        )

    # 3. Create separate summary plots for validation warnings and validation ok rate.
        
    # Validation warning counts
    w_means = [float(warnings_stats[n][0] or 0.0) for n in names_in_order]
    w_stds = [float(warnings_stats[n][1] or 0.0) for n in names_in_order]
    _bar_with_error(
        x_labels=x_labels,
        means=w_means,
        stds=w_stds,
        title=f"Validation warnings per setting (mean ± std across N={n_patients} patients)",
        ylabel="Warnings (count)",
        out_path=fig_dir / "warnings_mean_std.png",
        clamp01=False,
    )

    # Validation ok rate
    ok_means = [float(ok_rate_stats[n][0] or 0.0) for n in names_in_order]
    ok_stds = [float(ok_rate_stats[n][1] or 0.0) for n in names_in_order]
    _bar_with_error(
        x_labels=x_labels,
        means=ok_means,
        stds=ok_stds,
        title=f"Validation OK rate per setting (mean ± std across N={n_patients} patients)",
        ylabel="OK rate",
        out_path=fig_dir / "validation_ok_rate_mean_std.png",
        clamp01=True,
    )

    # 4. Save the aggregated metrics table for reporting and thesis use.
    _write_aggregate_csv(
        out_path=fig_dir / "multi_patient_aggregate_table.csv",
        names_in_order=names_in_order,
        x_labels=x_labels,
        stats={name: stats[name] for name in names_in_order},
        n_patients_by_name=n_patients_by_name,
        mean_warnings=warnings_stats,
        ok_rate=ok_rate_stats,
    )

    print("\n✅ Multi-patient figures saved to:")
    print(" -", str(fig_dir))
    print("Figures:")
    print(" - weights_stacked.png")
    for block_name, _, _ in BLOCKS:
        print(f" - {block_name}.png")
    print(" - warnings_mean_std.png")
    print(" - validation_ok_rate_mean_std.png")
    print("\nTable:")
    print(" - multi_patient_aggregate_table.csv")
    print("\nUsed experiment folder:")
    print(" -", str(exp_dir))


if __name__ == "__main__":
    main()
