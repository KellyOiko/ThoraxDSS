"""
Utility functions for text-based evaluation metrics such as BLEU, ROUGE, METEOR, chrF, and BERTScore.
"""
import re
from collections import Counter
from typing import Any, Dict, Optional, List


# Small text helpers

def strip_xxxx(sent_list: List[str]) -> List[str]:
    """Remove placeholder tokens such as XXXX from a list of sentences."""
    return [s.replace("XXXX", "").strip() for s in sent_list if isinstance(s, str)]

def flatten_sentences(sent_list: List[str]) -> str:
    """Join a sentence list into a single text string after placeholder cleanup."""
    return " ".join(strip_xxxx(sent_list))

def normalize_text(s: str) -> str:
    """Lowercase and normalize whitespace in text."""
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


# Metric computation helpers

def compute_bleu(pred: str, ref: str) -> Optional[float]:
    """Compute sentence-level BLEU using sacrebleu when available."""
    try:
        import sacrebleu
        return float(sacrebleu.sentence_bleu(pred, [ref]).score)
    except Exception:
        return None

def _tokenize(text: str):
    """Tokenize text into simple lowercase alphanumeric tokens."""
    text = (text or "").lower()
    return re.findall(r"[a-z0-9]+", text)

def _ngrams(tokens, n):
    """Build token n-grams from a token sequence."""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def _f1(p, r):
    """Compute harmonic-mean F1 from precision and recall."""
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def _rouge_n_f1(pred: str, ref: str, n: int) -> float:
    """Compute ROUGE-N-style F1 from token n-gram overlap."""
    ptoks = _tokenize(pred)
    rtoks = _tokenize(ref)

    png = Counter(_ngrams(ptoks, n))
    rng = Counter(_ngrams(rtoks, n))

    if not png or not rng:
        return 0.0

    overlap = sum((png & rng).values())
    p = overlap / max(sum(png.values()), 1)
    r = overlap / max(sum(rng.values()), 1)
    return _f1(p, r)

def _lcs_length(a, b):
    """Compute the longest common subsequence length between two token lists."""
    m, n = len(a), len(b)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = temp
    return dp[n]

def _rouge_l_f1(pred: str, ref: str) -> float:
    """Compute ROUGE-L-style F1 from longest common subsequence overlap."""
    ptoks = _tokenize(pred)
    rtoks = _tokenize(ref)

    if not ptoks or not rtoks:
        return 0.0

    lcs = _lcs_length(ptoks, rtoks)
    p = lcs / max(len(ptoks), 1)
    r = lcs / max(len(rtoks), 1)
    return _f1(p, r)

def compute_rouge_light(pred: str, ref: str) -> Dict[str, float]:
    """Compute lightweight ROUGE-style F1 scores without external ROUGE dependencies."""
    return {
        "rouge1_f": _rouge_n_f1(pred, ref, 1),
        "rouge2_f": _rouge_n_f1(pred, ref, 2),
        "rougeL_f": _rouge_l_f1(pred, ref),
    }

def compute_rouge(pred: str, ref: str) -> Dict[str, float]:
    """Compute ROUGE scores, falling back to a lightweight internal implementation if needed."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(ref, pred)
        return {
            "rouge1_f": float(scores["rouge1"].fmeasure),
            "rouge2_f": float(scores["rouge2"].fmeasure),
            "rougeL_f": float(scores["rougeL"].fmeasure),
        }
    except Exception:
        return compute_rouge_light(pred, ref)

def compute_meteor(pred: str, ref: str) -> Optional[float]:
    """
    Compute METEOR when NLTK meteor support is available.

    NLTK>=3.9 expects pre-tokenized inputs.
    """
    try:
        from nltk.translate.meteor_score import meteor_score
        # simple tokenization; keeps it dependency-light
        pred_toks = _tokenize(pred or "")
        ref_toks  = _tokenize(ref or "")
        if not pred_toks or not ref_toks:
            return 0.0
        return float(meteor_score([ref_toks], pred_toks))
    except Exception:
        return None

def compute_chrf(pred: str, ref: str) -> Optional[float]:
    """Compute sentence-level chrF using sacrebleu when available."""
    try:
        import sacrebleu
        return float(sacrebleu.sentence_chrf(pred, [ref]).score)
    except Exception:
        return None

def compute_bertscore(pred: str, ref: str) -> Optional[float]:
    """Compute BERTScore F1 using a lightweight English encoder when available."""
    try:
        from bert_score import score

        P, R, F1 = score(
            [pred], [ref],
            lang="en",
            model_type="distilbert-base-uncased",
            verbose=False,
            device="cpu",
        )

        return float(F1[0].item())
    except Exception:
        return None

def compute_text_metrics(pred: str, ref: str) -> Dict[str, Any]:
    """Compute the full set of text-based evaluation metrics for a prediction/reference pair."""
    pred = pred or ""
    ref = ref or ""

    metrics: Dict[str, Any] = {}
    metrics["bleu"] = compute_bleu(pred, ref)
    metrics.update(compute_rouge(pred, ref))
    metrics["meteor"] = compute_meteor(pred, ref)
    metrics["chrf"] = compute_chrf(pred, ref)
    metrics["bertscore_f1"] = compute_bertscore(pred, ref)
    return metrics
