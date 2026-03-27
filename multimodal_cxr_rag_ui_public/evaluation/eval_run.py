"""
Evaluation utilities for structured multimodal RAG outputs.

This module evaluates:
- patient findings generation,
- retrieved case findings,
- weak-label similarity classification,
- comparison-summary concept overlap,
- final diagnosis assessment consistency.
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Set
from sklearn.metrics.pairwise import cosine_similarity

# local metrics
from evaluation.text_metrics import (
    compute_text_metrics,
    flatten_sentences,
    normalize_text,
)

# Small helper to keep eval_run self-contained.
def safe_get_list(d: dict, key: str) -> List[str]:
    """Return a list value from a dictionary, or an empty list if the key is missing or invalid."""
    v = d.get(key, [])
    return v if isinstance(v, list) else []

# 2) Semantic similarity (weak labels)

_SIM_MODEL = None

def _get_sim_model():
    """
    Lazy-load SentenceTransformer once (avoids heavy import cost on module import).
    """
    global _SIM_MODEL
    if _SIM_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SIM_MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _SIM_MODEL

def embed_similarity(pred: str, ref: str) -> Optional[float]:
    """
    Compute cosine similarity between MiniLM sentence embeddings.
    Returns None if either text is empty or if the embedding model is unavailable.
    """
    if not pred or not ref:
        return None
    try:
        model = _get_sim_model()
        emb = model.encode([pred, ref], normalize_embeddings=True)
        sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
        return float(sim)
    except Exception:
        return None

def weak_label_from_similarity(
    sim: float,
    thr_similar: float = 0.70,
    thr_partial: float = 0.45
) -> str:
    """Convert a continuous similarity score into a weak similarity label."""
    if sim >= thr_similar:
        return "similar"
    if sim >= thr_partial:
        return "partially similar"
    return "not similar"

# 3) Simple findings extractor for D (silver sets)

MED_KEYWORDS = [
    "pneumothorax", "effusion", "atelectasis", "consolidation",
    "opacity", "infiltrate", "edema", "cardiomegaly",
    "hyperlucent", "hyperexpanded", "emphysema",
    "mediastinal shift", "tracheal shift", "pleural air",
    "lung collapse", "collapse",
]

def extract_findings_set_from_text(text: str) -> Set[str]:
    """Extract a lightweight keyword-based finding set from free text."""
    t = normalize_text(text)
    concepts: Set[str] = set()
    for kw in MED_KEYWORDS:
        if kw in t:
            concepts.add(kw)
    return concepts

def extract_findings_set_from_sentences(sents: List[str]) -> Set[str]:
    """Extract a keyword-based finding set from a list of sentences."""
    return extract_findings_set_from_text(" ".join([s for s in sents if isinstance(s, str)]))

def set_precision_recall_f1(pred_set: Set[str], gold_set: Set[str]) -> Dict[str, float]:
    """Compute precision, recall, and F1 for predicted versus reference concept sets."""
    pred_set = set(pred_set)
    gold_set = set(gold_set)

    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = len(pred_set & gold_set)
    precision = tp / max(len(pred_set), 1)
    recall = tp / max(len(gold_set), 1)
    f1 = 0.0 if (precision + recall) == 0 else (2 * precision * recall / (precision + recall))
    return {"precision": precision, "recall": recall, "f1": f1}


# 4) A) Patient findings eval
def eval_patient_findings(j: dict, patient_report_text: str) -> Dict[str, Any]:
    """Evaluate generated patient findings against the patient report text."""
    pred_text = flatten_sentences(safe_get_list(j, "patient_findings"))
    ref_text = patient_report_text

    return {
        "pred_text": pred_text,
        "ref_text": ref_text,
        "text_metrics": compute_text_metrics(pred_text, ref_text),
        "pred_concepts": sorted(list(extract_findings_set_from_sentences(safe_get_list(j, "patient_findings")))),
        "ref_concepts": sorted(list(extract_findings_set_from_text(patient_report_text))),
    }

# 5) B) Retrieved case findings eval
def eval_retrieved_cases(j: dict, case_reports: List[str]) -> List[Dict[str, Any]]:
    """Evaluate generated retrieved-case findings against the corresponding case reports."""
    out: List[Dict[str, Any]] = []
    retrieved = j.get("retrieved_cases", [])
    if not isinstance(retrieved, list):
        return out

    n = min(len(retrieved), len(case_reports))

    for i in range(n):
        rc = retrieved[i] or {}
        pred_sents = safe_get_list(rc, "findings")
        pred_text = flatten_sentences(pred_sents)
        ref_text = case_reports[i]

        out.append({
            "case_index": rc.get("case_index", i + 1),
            "pred_text": pred_text,
            "ref_text": ref_text,
            "text_metrics": compute_text_metrics(pred_text, ref_text),
            "pred_concepts": sorted(list(extract_findings_set_from_sentences(pred_sents))),
            "ref_concepts": sorted(list(extract_findings_set_from_text(ref_text))),
        })
    return out

# 6) C) similarity_to_patient (weak label classification)
def eval_similarity_classification(
    j: dict,
    patient_report_text: str,
    case_reports: List[str],
    thr_similar: float = 0.70,
    thr_partial: float = 0.45
) -> Dict[str, Any]:
    """
    Evaluate similarity_to_patient as a weak-label classification task.

    Gold labels are derived from embedding similarity between the patient report
    and each retrieved case report.
    """
    retrieved = j.get("retrieved_cases", [])
    if not isinstance(retrieved, list):
        return {"supported": False, "reason": "retrieved_cases not a list"}

    y_pred: List[str] = []
    y_gold: List[str] = []
    sims: List[float] = []

    n = min(len(retrieved), len(case_reports))

    for i in range(n):
        rc = retrieved[i] or {}

        pred_label = rc.get("similarity_to_patient")
        pred_label = pred_label.strip().lower() if isinstance(pred_label, str) else "unknown"

        sim = embed_similarity(patient_report_text, case_reports[i])
        if sim is None:
            return {"supported": False, "reason": "sentence-transformers not available"}

        gold_label = weak_label_from_similarity(sim, thr_similar, thr_partial)

        y_pred.append(pred_label)
        y_gold.append(gold_label)
        sims.append(sim)

    try:
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        acc = float(accuracy_score(y_gold, y_pred))
        macro_f1 = float(f1_score(y_gold, y_pred, average="macro"))
        report = classification_report(y_gold, y_pred, output_dict=True, zero_division=0)
    except Exception:
        acc = None
        macro_f1 = None
        report = None

    return {
        "supported": True,
        "thresholds": {"similar": thr_similar, "partial": thr_partial},
        "similarities": sims,
        "y_gold": y_gold,
        "y_pred": y_pred,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "classification_report": report,
    }

# 7) D) Comparison summary eval (silver gold)
from backend.negation import TermProcessor
from evaluation.concepts import extract_present_terms, extract_present_terms_from_sentences

def eval_comparison_summary(
    j: dict,
    patient_report_text: str,
    case_reports: List[str],
    tp: TermProcessor,
) -> List[Dict[str, Any]]:
    """
    Evaluate comparison-summary outputs using silver concept sets derived from text.

    Gold common/difference sets are built from present terms extracted from the
    patient and case reports using the shared project vocabulary.
    """    
    out: List[Dict[str, Any]] = []

    cs_list = j.get("comparison_summary", [])
    if not isinstance(cs_list, list):
        return out

    # Build silver gold sets from text using the same vocabulary and synonym resources as the project.
    P = extract_present_terms(patient_report_text, tp)

    n = min(len(cs_list), len(case_reports))
    for i in range(n):
        cs = cs_list[i] or {}
        case_text = case_reports[i]

        C = extract_present_terms(case_text, tp)

        gold_common = P & C
        gold_diff = C - P

        pred_common_sents = safe_get_list(cs, "common_with_patient")
        pred_diff_sents = safe_get_list(cs, "differences_from_patient")

        pred_common_set = extract_present_terms_from_sentences(pred_common_sents, tp)
        pred_diff_set = extract_present_terms_from_sentences(pred_diff_sents, tp)

        common_scores = set_precision_recall_f1(pred_common_set, gold_common)
        diff_scores = set_precision_recall_f1(pred_diff_set, gold_diff)

        out.append({
            "case_index": cs.get("case_index", i + 1),
            "gold_common": sorted(list(gold_common)),
            "gold_diff": sorted(list(gold_diff)),
            "pred_common": sorted(list(pred_common_set)),
            "pred_diff": sorted(list(pred_diff_set)),
            "common_set_metrics": common_scores,
            "diff_set_metrics": diff_scores,
            "pred_common_sents": pred_common_sents,
            "pred_diff_sents": pred_diff_sents,
        })

    return out

# 8) E) Final diagnosis assessment eval

def eval_final_diagnosis(j: dict, patient_report_text: str) -> Dict[str, Any]:
    """Evaluate the final diagnosis assessment text against the patient report text."""
    fd = j.get("final_diagnosis_assessment", {}) if isinstance(j, dict) else {}
    pred_text = fd.get("text", "")
    if not isinstance(pred_text, str):
        pred_text = ""

    pred_text = pred_text.replace("XXXX", "").strip()
    ref_text = patient_report_text

    return {
        "pred_text": pred_text,
        "ref_text": ref_text,
        "text_metrics": compute_text_metrics(pred_text, ref_text),
    }

# 9) Master/Main function (single call)
from evaluation.concepts import build_term_processor

def evaluate_run(
    out: dict,
    patient_report_text: str,
    thr_similar: float = 0.70,
    thr_partial: float = 0.45
) -> Dict[str, Any]:
    """
    Run the full evaluation pipeline for one structured system output.

    This function evaluates all major output components and returns a nested
    dictionary with per-stage evaluation results.
    """
        
    j = out.get("llm_response_structured")
    case_reports = out.get("case_reports", []) or []

    if j is None or not isinstance(j, dict):
        return {"ok": False, "reason": "No structured JSON"}

    results: Dict[str, Any] = {"ok": True}

    # Build the shared term processor once so concept-based evaluation uses the project vocabulary.
    tp = build_term_processor()

    results["patient_findings_eval"] = eval_patient_findings(j, patient_report_text)
    results["retrieved_cases_eval"] = eval_retrieved_cases(j, case_reports)
    results["similarity_classification_eval"] = eval_similarity_classification(
        j, patient_report_text, case_reports,
        thr_similar=thr_similar, thr_partial=thr_partial
    )
    results["comparison_summary_eval"] = eval_comparison_summary(
        j, patient_report_text, case_reports, tp
    )
    results["final_diagnosis_eval"] = eval_final_diagnosis(j, patient_report_text)

    return results




# ---------------------------
def mean_safe(xs):
    """Return the mean of numeric values only, or None if no numeric values are available."""
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


def summarize_results(results: dict) -> dict:
    """Build a compact summary dashboard from the full evaluation output."""

    A = results.get("patient_findings_eval", {}).get("text_metrics", {})
    B_list = results.get("retrieved_cases_eval", [])
    E = results.get("final_diagnosis_eval", {}).get("text_metrics", {})

    B_metrics = [c.get("text_metrics", {}) for c in B_list]

    mean_B_rougeL = mean_safe([m.get("rougeL_f") for m in B_metrics])
    mean_B_rouge1 = mean_safe([m.get("rouge1_f") for m in B_metrics])
    mean_B_bleu   = mean_safe([m.get("bleu") for m in B_metrics])
    mean_B_chrf   = mean_safe([m.get("chrf") for m in B_metrics])
    mean_B_berts  = mean_safe([m.get("bertscore_f1") for m in B_metrics])

    C = results.get("similarity_classification_eval", {})

    D_list = results.get("comparison_summary_eval", [])
    mean_common_f1 = mean_safe([x.get("common_set_metrics", {}).get("f1") for x in D_list])
    mean_diff_f1   = mean_safe([x.get("diff_set_metrics", {}).get("f1") for x in D_list])

    return {
        # A: patient
        "patient_bleu": A.get("bleu"),
        "patient_rougeL": A.get("rougeL_f"),
        "patient_chrf": A.get("chrf"),
        "patient_bertscore": A.get("bertscore_f1"),

        # B: mean over retrieved cases
        "mean_case_bleu": mean_B_bleu,
        "mean_case_rouge1": mean_B_rouge1,
        "mean_case_rougeL": mean_B_rougeL,
        "mean_case_chrf": mean_B_chrf,
        "mean_case_bertscore": mean_B_berts,

        # C: label quality (weak)
        "similarity_accuracy": C.get("accuracy"),
        "similarity_macro_f1": C.get("macro_f1"),

        # D: comparison (silver trend)
        "mean_common_f1": mean_common_f1,
        "mean_diff_f1": mean_diff_f1,

        # E: final assessment consistency proxy
        "final_bleu": E.get("bleu"),
        "final_rougeL": E.get("rougeL_f"),
        "final_chrf": E.get("chrf"),
        "final_bertscore": E.get("bertscore_f1"),
    }


def print_summary(summary: dict):
    """Print a compact formatted evaluation summary."""
    def f(x):
        return "None" if x is None else f"{x:.4f}" if isinstance(x, float) else str(x)

    keys = [
        "patient_bleu",     
        "patient_rougeL", "patient_chrf", "patient_bertscore",
        "mean_case_bleu",    
        "mean_case_rougeL", "mean_case_chrf", "mean_case_bertscore",
        "similarity_accuracy", "similarity_macro_f1",
        "mean_common_f1", "mean_diff_f1",
        "final_bleu",        
        "final_rougeL", "final_chrf", "final_bertscore"
    ]

    for k in keys:
        print(f"{k:>22}: {f(summary.get(k))}")
