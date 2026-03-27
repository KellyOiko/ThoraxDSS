"""
Post-hoc aggregation utilities for multi-patient retrieval-weight sweeps.

Reconstructs per-run metrics (including METEOR), builds a long CSV (patient × setting),
and computes aggregated summaries per retrieval-weight configuration.
"""
from __future__ import annotations

from pathlib import Path
import sys, os, json, csv
from collections import defaultdict
from typing import Any, Dict, List, Optional


# Force ASCII project root (FAISS cannot reliably read Unicode paths on Windows)
PREFERRED_ROOT = Path(r"C:\mm_cxr")  # <-- ASCII path
if not (PREFERRED_ROOT / "backend" / "main.py").exists():
    raise RuntimeError(
        f"Expected ASCII clone at {PREFERRED_ROOT} but backend/main.py not found.\n"
        f"Either clone/copy your project to C:\\mm_cxr, or change PREFERRED_ROOT."
    )

PROJECT_ROOT = PREFERRED_ROOT
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


# Input/output paths
IN_DIR  = PROJECT_ROOT / "evaluation" / "_multi_patient_sweep" / "n10"
OUT_DIR = IN_DIR  # write outputs next to existing files

PAT_GLOB = "patient_*.json"


# Helper utilities for safe aggregation and nested access
def mean_safe(xs: List[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if isinstance(x, (int, float))]
    return (sum(vals) / len(vals)) if vals else None

def safe_get(d: Dict[str, Any], *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def count_warnings(validation: Dict[str, Any]) -> int:
    return len(validation.get("warnings", []) or [])

def count_errors(validation: Dict[str, Any]) -> int:
    return len(validation.get("errors", []) or [])


# Reconstruct scalar summary metrics (extended with METEOR)
def build_summary_v2(run: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rebuild the scalar summary used in single-case sweeps, including METEOR.

    Combines information from:
    - results (preferred source)
    - summary (fallback when needed)
    - validation (for flags)

    Produces metrics grouped as:
    A: patient-level text metrics
    B: retrieved-case averages
    C: similarity classification scores
    D: concept-set overlap metrics
    E: final diagnosis consistency
    """

    results = run.get("results", {}) or {}
    summary_old = run.get("summary", {}) or {}
    validation = run.get("validation", {}) or {}

    # A) Patient-level metrics
    A_tm = safe_get(results, "patient_findings_eval", "text_metrics", default={}) or {}
    patient_bleu      = A_tm.get("bleu", summary_old.get("patient_bleu"))
    patient_rougeL    = A_tm.get("rougeL_f", summary_old.get("patient_rougeL"))
    patient_meteor    = A_tm.get("meteor", None)  # NEW
    patient_chrf      = A_tm.get("chrf", summary_old.get("patient_chrf"))
    patient_bertscore = A_tm.get("bertscore_f1", summary_old.get("patient_bertscore"))

    # B) Retrieved-case metrics
    B_list = results.get("retrieved_cases_eval", []) or []
    B_tms = [ (c.get("text_metrics", {}) or {}) for c in B_list if isinstance(c, dict) ]

    mean_case_bleu      = mean_safe([m.get("bleu") for m in B_tms])      if B_tms else summary_old.get("mean_case_bleu")
    mean_case_rougeL    = mean_safe([m.get("rougeL_f") for m in B_tms])  if B_tms else summary_old.get("mean_case_rougeL")
    mean_case_meteor    = mean_safe([m.get("meteor") for m in B_tms])    if B_tms else None  # NEW
    mean_case_chrf      = mean_safe([m.get("chrf") for m in B_tms])      if B_tms else summary_old.get("mean_case_chrf")
    mean_case_bertscore = mean_safe([m.get("bertscore_f1") for m in B_tms]) if B_tms else summary_old.get("mean_case_bertscore")

    # C) Similarity classification
    C = results.get("similarity_classification_eval", {}) or {}
    similarity_accuracy  = C.get("accuracy", summary_old.get("similarity_accuracy"))
    similarity_macro_f1  = C.get("macro_f1", summary_old.get("similarity_macro_f1"))

    # D) Concept overlap metrics
    D_list = results.get("comparison_summary_eval", []) or []
    mean_common_f1 = mean_safe([ (x.get("common_set_metrics", {}) or {}).get("f1") for x in D_list ]) if D_list else summary_old.get("mean_common_f1")
    mean_diff_f1   = mean_safe([ (x.get("diff_set_metrics", {}) or {}).get("f1") for x in D_list ])   if D_list else summary_old.get("mean_diff_f1")

    # E) Final diagnosis metrics
    E_tm = safe_get(results, "final_diagnosis_eval", "text_metrics", default={}) or {}
    final_bleu      = E_tm.get("bleu", summary_old.get("final_bleu"))
    final_rougeL    = E_tm.get("rougeL_f", summary_old.get("final_rougeL"))
    final_meteor    = E_tm.get("meteor", None)  # NEW
    final_chrf      = E_tm.get("chrf", summary_old.get("final_chrf"))
    final_bertscore = E_tm.get("bertscore_f1", summary_old.get("final_bertscore"))

    # Validation flags
    n_warnings = count_warnings(validation)
    n_errors   = count_errors(validation)
    validation_ok = bool(validation.get("ok", False))
    logic_ok      = bool(validation.get("logic_ok", False))

    return {
        # A
        "patient_bleu": patient_bleu,
        "patient_rougeL": patient_rougeL,
        "patient_meteor": patient_meteor,
        "patient_chrf": patient_chrf,
        "patient_bertscore": patient_bertscore,
        # B
        "mean_case_bleu": mean_case_bleu,
        "mean_case_rougeL": mean_case_rougeL,
        "mean_case_meteor": mean_case_meteor,
        "mean_case_chrf": mean_case_chrf,
        "mean_case_bertscore": mean_case_bertscore,
        # C
        "similarity_accuracy": similarity_accuracy,
        "similarity_macro_f1": similarity_macro_f1,
        # D
        "mean_common_f1": mean_common_f1,
        "mean_diff_f1": mean_diff_f1,
        # E
        "final_bleu": final_bleu,
        "final_rougeL": final_rougeL,
        "final_meteor": final_meteor,
        "final_chrf": final_chrf,
        "final_bertscore": final_bertscore,
        # Flags
        "n_warnings": n_warnings,
        "n_errors": n_errors,
        "validation_ok": validation_ok,
        "logic_ok": logic_ok,
    }

# Aggregate metrics across patients for each retrieval setting
def aggregate_across_patients(per_patient_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Input: list of dicts with keys: patient_id, name, weights, summary_v2
    Output: dict[name -> {n_patients, weights, mean_metrics}]
    """
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in per_patient_runs:
        grouped[r["name"]].append(r)

    out: Dict[str, Any] = {}
    for name, rows in grouped.items():
        # Assume weights are constant per name; take first
        weights = rows[0].get("weights", {}) if rows else {}

        # Collect numeric metrics
        metric_lists: Dict[str, List[float]] = defaultdict(list)
        bool_lists: Dict[str, List[bool]] = defaultdict(list)

        for row in rows:
            sv2 = row.get("summary_v2", {}) or {}
            for k, v in sv2.items():
                if isinstance(v, (int, float)):
                    metric_lists[k].append(float(v))
                elif isinstance(v, bool):
                    bool_lists[k].append(bool(v))

        mean_metrics = {k: mean_safe(vs) for k, vs in metric_lists.items()}
        
        # Convert boolean flags to rates
        for k, bs in bool_lists.items():
            if bs:
                mean_metrics[k] = sum(1 for x in bs if x) / len(bs)

        out[name] = {
            "n_patients": len(rows),
            "weights": weights,
            "mean_metrics": mean_metrics,
        }

    return out

def canonical_setting_key(weights: Dict[str, Any]) -> str:
    """Return a stable key in T2T–T2I–I2I order."""
    t2t = float(weights.get("t2t", 0.0) or 0.0)
    t2i = float(weights.get("t2i", 0.0) or 0.0)
    i2i = float(weights.get("i2i", 0.0) or 0.0)
    return f"T2T={t2t:.2f} T2I={t2i:.2f} I2I={i2i:.2f}"




def main():
    patient_files = sorted(IN_DIR.glob(PAT_GLOB))
    if not patient_files:
        raise FileNotFoundError(f"No files found: {IN_DIR / PAT_GLOB}")

    per_patient_rows: List[Dict[str, Any]] = []

    for pf in patient_files:
        patient_id = pf.stem  # patient_001
        data = json.loads(pf.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            continue

        for run in data:
            if not isinstance(run, dict):
                continue
            weights = run.get("weights", {}) or {}
            summary_v2 = build_summary_v2(run)

            per_patient_rows.append({
                "patient_id": patient_id,
                "name": canonical_setting_key(weights),   # <-- IMPORTANT
                "weights": weights,
                "summary_v2": summary_v2,
            })

    # Write long-format CSV (patient × setting)
    csv_path = OUT_DIR / "multi_patient_weight_sweep_long_v2.csv"
    
    cols = [
        "patient_id", "name", "w_t2t", "w_t2i", "w_i2i",
        "patient_bleu", "patient_rougeL", "patient_meteor", "patient_chrf", "patient_bertscore",
        "mean_case_bleu", "mean_case_rougeL", "mean_case_meteor", "mean_case_chrf", "mean_case_bertscore",
        "similarity_accuracy", "similarity_macro_f1", "mean_common_f1", "mean_diff_f1",
        "final_bleu", "final_rougeL", "final_meteor", "final_chrf", "final_bertscore",
        "n_warnings", "n_errors", "validation_ok", "logic_ok",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in per_patient_rows:
            weights = r.get("weights", {}) or {}
            sv2 = r.get("summary_v2", {}) or {}
            row = {
                "patient_id": r["patient_id"],
                "name": r["name"],
                "w_t2t": weights.get("t2t"),
                "w_t2i": weights.get("t2i"),
                "w_i2i": weights.get("i2i"),
            }
            row.update(sv2)
            w.writerow(row)

    # Aggregate means per setting
    aggregated = aggregate_across_patients(per_patient_rows)
    out_path = OUT_DIR / "multi_patient_weight_sweep_summary_v2.json"
    out_path.write_text(json.dumps(aggregated, indent=2), encoding="utf-8")

    print("✅ Wrote:", csv_path)
    print("✅ Wrote:", out_path)

if __name__ == "__main__":
    main()
