"""
Concept-extraction helpers for evaluation using the project's negation-aware term processor.
"""
from __future__ import annotations
from typing import Iterable, Optional, Set
from backend.negation import TermConfig, TermProcessor


def build_term_processor(vocab_xlsx_path: Optional[str] = None) -> TermProcessor:
    """
    Build a term processor using the project vocabulary and synonym resources.

    If vocab_xlsx_path is not provided, the vocabulary file is auto-detected.
    """
    cfg = TermConfig(vocab_xlsx_path=vocab_xlsx_path)
    return TermProcessor(cfg)


def extract_present_terms(text: str, tp: TermProcessor) -> Set[str]:
    """
    Canonical terms that are PRESENT (polarity +1) in the text.
    Synonyms are mapped back to canonical automatically.
    """
    m = tp.build_polarity_map(text or "")
    return {t for t, pol in m.items() if pol == +1}


def extract_present_terms_from_sentences(sents: Iterable[str], tp: TermProcessor) -> Set[str]:
    """Extract canonical present terms from a list of sentences."""
    joined = " ".join([str(s) for s in (sents or []) if isinstance(s, str) and s.strip()])
    return extract_present_terms(joined, tp)
