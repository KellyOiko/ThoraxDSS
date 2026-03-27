"""
Run retrieval-weight sweeps across multiple patients and aggregate the results.
"""
from pathlib import Path
import json
from collections import defaultdict

from backend.main import get_system
from evaluation.sweep_retrieval_weights import run_one_setting


def load_patients(df_sample, image_root: Path):
    """Build a patient list from df_sample, resolving report text and image paths robustly."""
    patients = []
    for _, row in df_sample.iterrows():
        report_id = str(row.get("report", ""))

        # Use the report text column expected by the main pipeline.
        report_text = str(row.get("text", ""))

        # Resolve the patient image path, preferring img_path when already available.
        if "img_path" in df_sample.columns and str(row.get("img_path", "")).strip():
            image_path = str(Path(row["img_path"]))
        else:
            # fallback to filename-style column
            fname = (
                row.get("image")
                or row.get("img")
                or row.get("filename")
            )
            if fname is None:
                raise KeyError("df_sample must contain either 'img_path' or one of: image/img/filename")
            image_path = str(Path(image_root) / str(fname))

        patients.append({
            "report_id": report_id,
            "report_text": report_text,
            "image_path": image_path,
        })
    return patients


def run_weight_sweep_core(
    system,
    query_text: str,
    patient_path: str,
    exclude_report: str,
    k_cases: int,
    model: str,
    settings: list,
):
    """
    Run all retrieval-weight settings for one patient/query and collect
    summaries, validation output, and detailed evaluation results.
    """
    sweep_results = []

    for name, w_t2i, w_i2i in settings:
        w_t2t = max(0.0, 1.0 - w_t2i - w_i2i)

        summary, results, out = run_one_setting(
            system,
            query_text,
            patient_path,
            k=k_cases,
            model=model,
            w_t2i=w_t2i,
            w_i2i=w_i2i,
            exclude_report=exclude_report,    # Leave-one-out: exclude the current patient report from retrieval.
        )

        validation = out.get("validation", {}) or {}

        sweep_results.append({
            "name": name,
            "weights": {"t2t": w_t2t, "t2i": w_t2i, "i2i": w_i2i},
            "summary": summary,
            "validation": validation,
            "results": results,
        })

    return sweep_results


def run_multi_patient_weight_sweep(
    df_sample,
    image_root: Path,
    out_dir: Path,
    k_cases: int,
    model: str,
):
    """
    Run the retrieval-weight sweep across multiple patients and save both
    per-patient outputs and aggregated summaries.
    """

    # Store this run under a fixed n10 subfolder for the current experiment setup.
    out_dir = Path(out_dir) / "n10" 
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the shared system once and reuse it across all patients/settings.
    system = get_system()   

    settings = [
        ("T2T=1.00 I2I=0.00 T2I=0.00", 0.00, 0.00),
        ("T2T=0.00 I2I=1.00 T2I=0.00", 0.00, 1.00),
        ("T2T=0.00 I2I=0.00 T2I=1.00", 1.00, 0.00),
        ("T2T=0.50 I2I=0.25 T2I=0.25", 0.25, 0.25),
        ("T2T=0.25 I2I=0.50 T2I=0.25", 0.25, 0.50),
        ("T2T=0.25 I2I=0.25 T2I=0.50", 0.50, 0.25),
    ]

    patients = load_patients(df_sample, image_root)


    # Restrict the sweep to the first N patients for a reproducible subset.
    N_PATIENTS = 10
    patients = patients[:N_PATIENTS]


    aggregate = defaultdict(list)

    # Run the full weight sweep for each selected patient.
    for idx, p in enumerate(patients, start=1):
        print(f"\n=== Patient {idx}/{len(patients)} | exclude={p['report_id']} ===")

        per_patient_results = run_weight_sweep_core(
            system=system,
            query_text=p["report_text"],
            patient_path=p["image_path"],
            exclude_report=p["report_id"],  # Leave-one-out: exclude the current patient report.
            k_cases=k_cases,
            model=model,
            settings=settings,
        )

        patient_out = out_dir / f"patient_{idx:03d}.json"
        patient_out.write_text(json.dumps(per_patient_results, indent=2), encoding="utf-8")

        # Group runs by weight-setting name so they can be aggregated across patients.
        for r in per_patient_results:
            aggregate[r["name"]].append(r)

    aggregated_summary = summarize_across_patients(aggregate)

    (out_dir / "multi_patient_weight_sweep_summary.json").write_text(
        json.dumps(aggregated_summary, indent=2),
        encoding="utf-8"
    )

    print("\n✅ Multi-patient sweep completed")
    print("✅ Output folder:", str(out_dir))



def summarize_across_patients(aggregate: dict):
    """
    Aggregate per-patient sweep results into mean metrics, warning counts,
    and validation success rates for each weight setting.
    """
    final = {}

    for name, runs in aggregate.items():
        metrics = defaultdict(list)
        ok_flags = []
        warn_counts = []

        for r in runs:
            s = r.get("summary", {}) or {}
            v = r.get("validation", {}) or {}

            # Collect numeric summary metrics so mean values can be computed across patients.
            for k, val in s.items():
                if isinstance(val, (int, float)):
                    metrics[k].append(float(val))

            # Collect validation status and warning counts for aggregate reporting.
            ok_flags.append(bool(v.get("ok", False)))
            warn_counts.append(len(v.get("warnings", []) or []))

        mean_metrics = {k: (sum(vals) / len(vals)) for k, vals in metrics.items() if vals}
        mean_warnings = (sum(warn_counts) / len(warn_counts)) if warn_counts else None
        ok_rate = (sum(1 for x in ok_flags if x) / len(ok_flags)) if ok_flags else None

        final[name] = {
            "n_patients": len(runs),
            "mean_metrics": mean_metrics,
            "mean_warnings": mean_warnings,
            "validation_ok_rate": ok_rate,
        }

    return final
