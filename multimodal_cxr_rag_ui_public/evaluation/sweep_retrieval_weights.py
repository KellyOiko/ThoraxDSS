"""
Run retrieval-weight sweeps and save both structured-output and raw-output evaluation results.
"""
from transformers import logging
logging.set_verbosity_error()

import json
from pathlib import Path
import hashlib

from backend.main import get_system
from evaluation.eval_run import evaluate_run, summarize_results
import re
from backend.validation import JSONParser  # uses your existing repair+parse


def set_weights(system, w_t2i: float, w_i2i: float):
    """
    Update retrieval weights both in the top-level system config and in the
    retriever instance, so subsequent runs use the same effective settings.
    """
        
    # 1) main.run_retrieval reads from system.config.retrieval
    system.config.retrieval.weight_text2image = float(w_t2i)
    system.config.retrieval.weight_image2image = float(w_i2i)

    # 2) retrieval.py reads from retriever config
    if hasattr(system.retriever, "config"):
        system.retriever.config.weight_text2image = float(w_t2i)
        system.retriever.config.weight_image2image = float(w_i2i)
    elif hasattr(system.retriever, "cfg"):
        system.retriever.cfg.weight_text2image = float(w_t2i)
        system.retriever.cfg.weight_image2image = float(w_i2i)


# evaluation/sweep_retrieval_weights.py

def run_one_setting(system, query_text, patient_path, k, model, w_t2i, w_i2i, exclude_report=None):
    """Run one retrieval-weight setting and return summary metrics, full results, and raw system output."""
    set_weights(system, w_t2i, w_i2i)
    out = system.run_multicase_analysis(
        query_text, patient_path, k=k, model=model, exclude_report=exclude_report
    )
    results = evaluate_run(out, patient_report_text=query_text)
    summary = summarize_results(results)
    return summary, results, out


def _slug(s: str) -> str:
    """Convert a setting name into a filesystem-safe slug."""
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")


def load_existing_full_dump(path: Path):
    """Load an existing JSON sweep dump if available; otherwise return an empty list."""
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def same_setting(entry: dict, name: str, w_t2i: float, w_i2i: float, w_t2t: float) -> bool:
    """Check whether a saved dump entry matches the current sweep setting."""
    if not isinstance(entry, dict):
        return False
    if entry.get("name") != name:
        return False
    weights = entry.get("weights", {}) or {}

    def close(a, b, eps=1e-9):
        try:
            return abs(float(a) - float(b)) <= eps
        except Exception:
            return False

    return (
        close(weights.get("t2i"), w_t2i)
        and close(weights.get("i2i"), w_i2i)
        and close(weights.get("t2t"), w_t2t)
    )


def dump_csv_from_dump(out_dir: Path, dump: list):
    """
    Write sweep_summary.csv from the entire dump (so CSV is complete even when runs are skipped).
    """
    csv_path = out_dir / "sweep_summary.csv"
    if not dump:
        csv_path.write_text("", encoding="utf-8")
        return

    # Flatten to rows
    rows = []
    for e in dump:
        s = (e.get("summary") or {})
        v = (e.get("validation") or {})
        w = (e.get("weights") or {})
        row = {
            "name": e.get("name"),
            "w_t2i": w.get("t2i"),
            "w_i2i": w.get("i2i"),
            "w_t2t": w.get("t2t"),
            "patient_bleu": s.get("patient_bleu"),
            "patient_rougeL": s.get("patient_rougeL"),
            "patient_chrf": s.get("patient_chrf"),
            "patient_bertscore": s.get("patient_bertscore"),
            "mean_case_bleu": s.get("mean_case_bleu"),
            "mean_case_rougeL": s.get("mean_case_rougeL"),
            "mean_case_chrf": s.get("mean_case_chrf"),
            "mean_case_bertscore": s.get("mean_case_bertscore"),
            "similarity_accuracy": s.get("similarity_accuracy"),
            "similarity_macro_f1": s.get("similarity_macro_f1"),
            "mean_common_f1": s.get("mean_common_f1"),
            "mean_diff_f1": s.get("mean_diff_f1"),
            "final_bleu": s.get("final_bleu"),
            "final_rougeL": s.get("final_rougeL"),
            "final_chrf": s.get("final_chrf"),
            "final_bertscore": s.get("final_bertscore"),
            "validation_ok": v.get("ok"),
            "n_warnings": len(v.get("warnings", []) or []),
        }
        rows.append(row)

    keys = list(rows[0].keys())
    lines = [",".join(keys)]
    for r in rows:
        vals = []
        for col in keys:
            v = r.get(col)
            if v is None:
                vals.append("")
            else:
                s = str(v)
                if "," in s or '"' in s:
                    s = '"' + s.replace('"', '""') + '"'
                vals.append(s)
        lines.append(",".join(vals))

    csv_path.write_text("\n".join(lines), encoding="utf-8")


def run_sweep_for_patient(
    system,
    query_text: str,
    patient_path: str,
    exclude_report: str,
    out_dir: Path,
    k_cases: int,
    model: str,
):
    """
    Run the full retrieval-weight sweep for one patient/example configuration.
    """

    #  experiment subfolder (deterministic)
    q_hash = hashlib.md5(query_text.encode("utf-8")).hexdigest()[:8]
    exp_name = f"pneumothorax_right__k{k_cases}__{model}__{q_hash}"
    out_dir = Path("evaluation/_sweeps") / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load system ONCE
    system = get_system()

    settings = [
        ("T2T=1.00 I2I=0.00 T2I=0.00", 0.00, 0.00),
        ("T2T=0.00 I2I=1.00 T2I=0.00", 0.00, 1.00),
        ("T2T=0.00 I2I=0.00 T2I=1.00", 1.00, 0.00),

        ("T2T=0.50 I2I=0.25 T2I=0.25", 0.25, 0.25),
        ("T2T=0.25 I2I=0.50 T2I=0.25", 0.25, 0.50),
        ("T2T=0.25 I2I=0.25 T2I=0.50", 0.50, 0.25),
    ]

    full_path = out_dir / "sweep_summary.json"
    existing_dump = load_existing_full_dump(full_path)

    for name, w_t2i, w_i2i in settings:
        w_t2t = max(0.0, 1.0 - w_t2i - w_i2i)
        slug = _slug(name)

        ui_path = out_dir / f"ui_struct__{slug}.json"
        dbg_path = out_dir / f"retrieval_debug__{slug}.json"

        already_in_summary = any(same_setting(e, name, w_t2i, w_i2i, w_t2t) for e in existing_dump)
        if ui_path.exists() and dbg_path.exists() and already_in_summary:
            print(f"[SKIP] {name} already computed.")
            continue

        print("\n" + "=" * 90)
        print(f"[RUN] {name}  requested: t2i={w_t2i} i2i={w_i2i}")
        print(f"[RUN] effective: t2t={w_t2t:.2f} i2i={w_i2i:.2f} t2i={w_t2i:.2f}")
        print("=" * 90)

        summary, results, out = run_one_setting(
            system, query_text, patient_path, k_cases, model, w_t2i=w_t2i, w_i2i=w_i2i
        )

        validation = out.get("validation", {}) or {}
        ui_struct = out.get("llm_response_structured", {}) or {}
        retrieval_debug = out.get("retrieval_debug", []) or []

        ui_path.write_text(json.dumps(ui_struct, ensure_ascii=False, indent=2), encoding="utf-8")
        dbg_path.write_text(json.dumps(retrieval_debug, ensure_ascii=False, indent=2), encoding="utf-8")

        existing_dump.append({
            "name": name,
            "weights": {"t2i": w_t2i, "i2i": w_i2i, "t2t": w_t2t},
            "summary": summary,
            "validation": validation,
            "results": results,
        })

        # incremental save
        full_path.write_text(json.dumps(existing_dump, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n[SUMMARY]", name)
        for metric_key in ["mean_case_rougeL", "mean_case_bertscore", "similarity_accuracy", "mean_common_f1", "mean_diff_f1"]:
            print(f"{metric_key:>22}: {summary.get(metric_key)}")
        print("[VALIDATION] ok=", validation.get("ok"), "warnings=", len(validation.get("warnings", []) or []))

    # final save + CSV rebuilt from whole dump
    full_path.write_text(json.dumps(existing_dump, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_csv_from_dump(out_dir, existing_dump)

    print("\n✅ Saved results to:")
    print(" -", str(out_dir / "sweep_summary.csv"))
    print(" -", str(out_dir / "sweep_summary.json"))
    print(" - per-setting UI structs: ui_struct__*.json")
    print(" - per-setting retrieval debug: retrieval_debug__*.json")
    print(" - experiment folder:", str(out_dir))





# Helpers for reconstructing a structured object from llm_response_raw.
_HDR_PAT = "=== PATIENT_ONLY ==="
_HDR_PAIR_RAW = "=== PAIRWISE_RAW ==="
_HDR_FINAL = "=== FINAL ==="

def _extract_section(raw: str, start_hdr: str, end_hdrs: list[str]) -> str:
    if not isinstance(raw, str):
        return ""
    s = raw
    start = s.find(start_hdr)
    if start < 0:
        return ""
    start += len(start_hdr)
    # find nearest end header after start
    end_positions = [s.find(h, start) for h in end_hdrs if s.find(h, start) >= 0]
    end = min(end_positions) if end_positions else len(s)
    return s[start:end].strip()

def _split_pairwise_items(pairwise_block: str) -> list[str]:
    if not isinstance(pairwise_block, str) or not pairwise_block.strip():
        return []
    # Your raw builder joins items with "\n\n---\n\n"
    parts = [p.strip() for p in pairwise_block.split("\n\n---\n\n") if p.strip()]
    return parts

def build_struct_from_llm_response_raw(out: dict) -> dict:
    """
    Rebuild a llm_response_structured-like object using only out["llm_response_raw"].

    This is used to evaluate the raw model output independently from the
    post-processed structured object returned by the main pipeline.
    """
    raw = out.get("llm_response_raw", "")
    n_cases = int(out.get("n_cases") or 0)

    patient_block = _extract_section(raw, _HDR_PAT, ["=== PAIRWISE_RAW ===", "=== PAIRWISE_POST ===", "=== FINAL ==="])
    pair_raw_block = _extract_section(raw, _HDR_PAIR_RAW, ["=== PAIRWISE_POST ===", "=== FINAL ==="])
    final_block = _extract_section(raw, _HDR_FINAL, [])  # until end

    patient_obj = JSONParser.try_parse_with_repair(patient_block)
    patient_obj = JSONParser.normalize_structure(patient_obj) if patient_obj else None
    patient_findings = []
    if isinstance(patient_obj, dict) and isinstance(patient_obj.get("patient_findings"), list):
        patient_findings = [str(x).strip() for x in patient_obj.get("patient_findings", []) if isinstance(x, str) and x.strip()]

    pair_items = _split_pairwise_items(pair_raw_block)
    retrieved_cases = []
    comparison_summary = []

    for i in range(1, n_cases + 1):
        item = pair_items[i - 1] if (i - 1) < len(pair_items) else ""
        pair_obj = JSONParser.try_parse_with_repair(item)
        pair_obj = JSONParser.normalize_structure(pair_obj) if pair_obj else None

        if not isinstance(pair_obj, dict):
            # fallback empty but aligned
            retrieved_cases.append({"case_index": i, "findings": [], "similarity_to_patient": "not similar"})
            comparison_summary.append({
                "case_index": i,
                "similarity_to_patient": "not similar",
                "common_with_patient": [],
                "differences_from_patient": [],
                "one_line_rationale": "",
            })
            continue

        cf = pair_obj.get("case_findings", [])
        cf = cf if isinstance(cf, list) else []
        cf = [str(x).strip() for x in cf if isinstance(x, str) and x.strip()]

        sim = pair_obj.get("similarity_to_patient", "not similar")
        sim = sim.strip().lower() if isinstance(sim, str) else "not similar"
        if sim not in {"similar", "partially similar", "not similar"}:
            sim = "not similar"

        common = pair_obj.get("common_with_patient", [])
        common = common if isinstance(common, list) else []
        common = [str(x).strip() for x in common if isinstance(x, str) and x.strip()]

        diffs = pair_obj.get("differences_from_patient", [])
        diffs = diffs if isinstance(diffs, list) else []
        diffs = [str(x).strip() for x in diffs if isinstance(x, str) and x.strip()]

        rationale = pair_obj.get("one_line_rationale", "")
        rationale = rationale.strip() if isinstance(rationale, str) else ""

        retrieved_cases.append({"case_index": i, "findings": cf, "similarity_to_patient": sim})
        comparison_summary.append({
            "case_index": i,
            "similarity_to_patient": sim,
            "common_with_patient": common,
            "differences_from_patient": diffs,
            "one_line_rationale": rationale,
        })

    # Final: try parse (may be nested)
    final_obj = JSONParser.try_parse_with_repair(final_block)
    final_obj = JSONParser.normalize_structure(final_obj) if final_obj else None
    if isinstance(final_obj, dict) and isinstance(final_obj.get("final_diagnosis_assessment"), dict):
        final_obj = final_obj["final_diagnosis_assessment"]

    final_text = ""
    final_conf = "low"
    if isinstance(final_obj, dict):
        t = final_obj.get("text", "")
        c = final_obj.get("confidence", "")
        if isinstance(t, str) and t.strip():
            final_text = t.strip()
        if isinstance(c, str) and c.strip().lower() in {"low", "medium", "high"}:
            final_conf = c.strip().lower()

    return {
        "patient_findings": patient_findings,
        "retrieved_cases": retrieved_cases,
        "comparison_summary": comparison_summary,
        "final_diagnosis_assessment": {"text": final_text, "confidence": final_conf},
    }

def run_one_setting_raw(system, query_text, patient_path, k, model, w_t2i, w_i2i, exclude_report=None):
    """
    Run one retrieval-weight setting and evaluate a structure rebuilt from llm_response_raw.
    """
    set_weights(system, w_t2i, w_i2i)
    out = system.run_multicase_analysis(
        query_text, patient_path, k=k, model=model, exclude_report=exclude_report
    )

    raw_struct = build_struct_from_llm_response_raw(out)

    # Make a shallow copy and override the structured payload used by evaluate_run()
    out_for_eval = dict(out)
    out_for_eval["llm_response_structured"] = raw_struct

    results = evaluate_run(out_for_eval, patient_report_text=query_text)
    summary = summarize_results(results)
    return summary, results, out, raw_struct


# Retrieval-weight settings to compare.
if __name__ == "__main__":
    from datetime import datetime

    query_text = (
        "Tension pneumothorax. Large right pneumothorax, with a markedly "
        "hyperlucent right hemithorax as a result. Note the collapsed right lung, "
        "and the tracheal and mediastinal shift. At the base of the right hemithorax "
        "is a small pleural effusion, but without a meniscus sign."
    )
    patient_path = "new_xrays/right-pneumothorax.png"
    k_cases = 3
    model = "minicpm-v"

    # unique folder each run
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")

    # STRUCTURED-eval folder (existing behavior)
    out_dir = Path("evaluation/_sweeps") / f"sweep_one_run_pneumothorax_k3__{model}__{run_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # RAW-eval folder (new)
    raw_out_dir = Path("evaluation/_sweeps") / f"sweep_one_run_pneumothorax_k3__RAW__{model}__{run_tag}"
    raw_out_dir.mkdir(parents=True, exist_ok=True)

    system = get_system()

    settings = [
        ("T2T=1.00 I2I=0.00 T2I=0.00", 0.00, 0.00),
        ("T2T=0.00 I2I=1.00 T2I=0.00", 0.00, 1.00),
        ("T2T=0.00 I2I=0.00 T2I=1.00", 1.00, 0.00),
        ("T2T=0.50 I2I=0.25 T2I=0.25", 0.25, 0.25),
        ("T2T=0.25 I2I=0.50 T2I=0.25", 0.25, 0.50),
        ("T2T=0.25 I2I=0.25 T2I=0.50", 0.50, 0.25),
    ]

    # Accumulators for structured-eval and raw-eval sweep outputs.

    # dumps
    existing_dump = []
    full_path = out_dir / "sweep_summary.json"

    raw_dump = []
    raw_full_path = raw_out_dir / "sweep_summary_raw.json"

    # Run each weight setting once for both structured and raw evaluation modes.
    for name, w_t2i, w_i2i in settings:
        w_t2t = max(0.0, 1.0 - w_t2i - w_i2i)
        slug = _slug(name)

        print("\n" + "=" * 90)
        print(f"[RUN] {name}  requested: t2i={w_t2i} i2i={w_i2i}")
        print(f"[RUN] effective: t2t={w_t2t:.2f} i2i={w_i2i:.2f} t2i={w_t2i:.2f}")
        print("=" * 90)

        # ===================== STRUCTURED EVAL =====================
        summary, results, out = run_one_setting(
            system, query_text, patient_path, k_cases, model, w_t2i=w_t2i, w_i2i=w_i2i
        )

        validation = out.get("validation", {}) or {}
        ui_struct = out.get("llm_response_structured", {}) or {}
        retrieval_debug = out.get("retrieval_debug", []) or []

        (out_dir / f"ui_struct__{slug}.json").write_text(
            json.dumps(ui_struct, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / f"retrieval_debug__{slug}.json").write_text(
            json.dumps(retrieval_debug, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        existing_dump.append({
            "name": name,
            "weights": {"t2i": w_t2i, "i2i": w_i2i, "t2t": w_t2t},
            "summary": summary,
            "validation": validation,
            "results": results,
        })
        full_path.write_text(json.dumps(existing_dump, ensure_ascii=False, indent=2), encoding="utf-8")

        # ===================== RAW EVAL (new) =====================
        raw_summary, raw_results, raw_out, raw_struct = run_one_setting_raw(
            system, query_text, patient_path, k_cases, model, w_t2i=w_t2i, w_i2i=w_i2i
        )

        raw_validation = raw_out.get("validation", {}) or {}
        raw_retrieval_debug = raw_out.get("retrieval_debug", []) or []

        (raw_out_dir / f"raw_struct__{slug}.json").write_text(
            json.dumps(raw_struct, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (raw_out_dir / f"llm_raw__{slug}.txt").write_text(
            raw_out.get("llm_response_raw", "") or "", encoding="utf-8"
        )
        (raw_out_dir / f"retrieval_debug__{slug}.json").write_text(
            json.dumps(raw_retrieval_debug, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        raw_dump.append({
            "name": name,
            "weights": {"t2i": w_t2i, "i2i": w_i2i, "t2t": w_t2t},
            "summary": raw_summary,
            "validation": raw_validation,
            "results": raw_results,
        })
        raw_full_path.write_text(json.dumps(raw_dump, ensure_ascii=False, indent=2), encoding="utf-8")

    # write CSVs
    dump_csv_from_dump(out_dir, existing_dump)
    dump_csv_from_dump(raw_out_dir, raw_dump)

    print("\n✅ Saved STRUCTURED-eval results to:")
    print(" -", str(out_dir / "sweep_summary.csv"))
    print(" -", str(out_dir / "sweep_summary.json"))
    print(" - per-setting UI structs: ui_struct__*.json")
    print(" - per-setting retrieval debug: retrieval_debug__*.json")
    print(" - experiment folder:", str(out_dir))

    print("\n✅ Saved RAW-eval results to:")
    print(" -", str(raw_out_dir / "sweep_summary.csv"))
    print(" -", str(raw_out_dir / "sweep_summary_raw.json"))
    print(" - per-setting RAW structs: raw_struct__*.json")
    print(" - per-setting raw text: llm_raw__*.txt")
    print(" - per-setting retrieval debug: retrieval_debug__*.json")
    print(" - RAW experiment folder:", str(raw_out_dir))
