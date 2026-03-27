"""
Corpus-level utility for computing negated and affirmed radiology term statistics.
"""
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple
import pandas as pd
from backend.config import AppConfig
from backend.negation import TermConfig, TermProcessor


def _resolve_project_path(p: Path | str) -> Path:
    """Resolve relative paths against project root (folder that contains /backend)."""
    project_root = Path(__file__).resolve().parent.parent  # .../multimodal_cxr_rag_ui
    p = Path(p)
    return p if p.is_absolute() else (project_root / p).resolve()

def _resolve_xlsx_path_from_config_or_fallback(cfg_path: Path | str) -> Path:
    """
    Resolve XLSX path robustly:
    1) try config path (relative to project root)
    2) try common fallbacks:
       - project_root/data/radiology_vocabulary_final.xlsx
       - project_root/data/raw/radiology_vocabulary_final.xlsx
       - project_root/backend/data/radiology_vocabulary_final.xlsx
       - project_root/backend/data/raw/radiology_vocabulary_final.xlsx
    """
    project_root = Path(__file__).resolve().parent.parent  # .../multimodal_cxr_rag_ui
    p = Path(cfg_path) if cfg_path else Path("")

    # 1) config path
    if str(p).strip():
        p1 = p if p.is_absolute() else (project_root / p).resolve()
        if p1.exists():
            return p1

    # 2) fallbacks (trying all known places)
    candidates = [
        project_root / "data" / "radiology_vocabulary_final.xlsx",
        project_root / "data" / "raw" / "radiology_vocabulary_final.xlsx",
        project_root / "backend" / "data" / "radiology_vocabulary_final.xlsx",
        project_root / "backend" / "data" / "raw" / "radiology_vocabulary_final.xlsx",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    # return "best guess" for error message
    return (project_root / p).resolve() if str(p).strip() else candidates[0].resolve()


def main():
    cfg = AppConfig()
    data = cfg.data

    # Resolve the main dataset and vocabulary paths from the application configuration.
    csv_path = _resolve_project_path(data.csv_path)         # Path
    df_sample_path = _resolve_project_path(data.df_sample_path)
    
    xlsx_path = _resolve_xlsx_path_from_config_or_fallback(getattr(data, "term_xlsx_path", ""))

    print("CSV PATH:", csv_path)
    print("DF_SAMPLE PATH:", df_sample_path)
    print("XLSX PATH:", xlsx_path)
    print("XLSX EXISTS:", xlsx_path.exists())

    # --- load dataset ---
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif df_sample_path.exists():
        # Fall back to df_sample.parquet when the full CSV is not available.
        df = pd.read_parquet(df_sample_path)
    else:
        raise FileNotFoundError(
            f"Neither CSV nor df_sample found.\nCSV: {csv_path}\nDF_SAMPLE: {df_sample_path}"
        )

    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column.")
    df["text"] = df["text"].fillna("").astype(str)

    # --- load vocab (terms + synonyms) ---
    if not xlsx_path.exists():
        raise FileNotFoundError(f"XLSX not found: {xlsx_path}")

    term_config = TermConfig(vocab_xlsx_path=str(xlsx_path))
    tp = TermProcessor(term_config)

    print("TERM LOAD WARNING:", getattr(term_config, "load_warning", ""))
    print("TERMS:", len(term_config.terms))
    print("SYN MAP:", len(getattr(term_config, "synonyms_map", {})))
    print("PHRASE VARIANTS:", len(getattr(tp, "phrase_variants", [])), "CANONICAL TERMS:", len(tp.terms))

    # --- corpus accumulators ---
    total_neg_mentions = defaultdict(int)
    total_pos_mentions = defaultdict(int)

    neg_reports = defaultdict(int)
    pos_reports = defaultdict(int)
    conflict_reports = defaultdict(int)

    # Iterate through report texts and accumulate mention / polarity statistics.
    for text in df["text"].tolist():
        neg_c, pos_c = count_mentions_in_text(tp, text)

        for t, c in neg_c.items():
            total_neg_mentions[t] += int(c)
        for t, c in pos_c.items():
            total_pos_mentions[t] += int(c)

        pol_map = tp.build_polarity_map(text)
        for term, pol in pol_map.items():
            if pol == -1:
                neg_reports[term] += 1
            elif pol == +1:
                pos_reports[term] += 1
            elif pol == 0:
                if (neg_c.get(term, 0) > 0) or (pos_c.get(term, 0) > 0):
                    conflict_reports[term] += 1

    # Build the final per-term statistics table.
    rows: List[Dict[str, Any]] = []
    for term in list(tp.terms):
        rows.append({
            "term": term,
            "neg_mentions": int(total_neg_mentions.get(term, 0)),
            "pos_mentions": int(total_pos_mentions.get(term, 0)),
            "neg_reports": int(neg_reports.get(term, 0)),
            "pos_reports": int(pos_reports.get(term, 0)),
            "conflict_reports": int(conflict_reports.get(term, 0)),
            "total_mentions": int(total_neg_mentions.get(term, 0) + total_pos_mentions.get(term, 0)),
        })

    out_df = pd.DataFrame(rows).sort_values(
        ["neg_mentions", "neg_reports", "total_mentions"],
        ascending=False
    ).reset_index(drop=True)

    # Save the output CSV into the configured processed-data directory.
    out_path = _resolve_project_path(data.processed_dir) / "negation_term_stats.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Saved stats -> {out_path}")
    print(out_df.head(20).to_string(index=False))



def count_mentions_in_text(tp: TermProcessor, text: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Count per-sentence term mentions.

    - neg_counts[term] += 1 when the term appears inside a negation scope
    - pos_counts[term] += 1 when the term appears in the sentence but outside negation scope

    This counts at most once per term, per sentence, per category (negated or affirmed),
    which is usually the most useful behavior for corpus statistics.
    """
    neg_counts = defaultdict(int)
    pos_counts = defaultdict(int)

    for sent in tp._split_sentences(text):
        tokens = tp._tokenize(sent)
        if not tokens:
            continue

        negated_in_sentence = set()

        # 1. Find negation scopes and count terms mentioned within them.
        i = 0
        while i < len(tokens):
            cue = None
            for c in tp.config.negation_cues:
                if tp._match_phrase_at(tokens, i, c):
                    cue = c
                    break

            if cue is None:
                i += 1
                continue

            scope_start = i + len(cue)

            # Skip list connectors immediately after the cue, e.g. "no ,", "no and".
            while scope_start < len(tokens) and tokens[scope_start] in tp.config.list_connectors:
                scope_start += 1

            scope_end = len(tokens)
            for j in range(scope_start, len(tokens)):
                if tokens[j] in tp.config.neg_scope_terminators:
                    scope_end = j
                    break

            neg_terms = tp._term_mentions_in_range(tokens, scope_start, scope_end)
            for term in neg_terms:
                neg_counts[term] += 1
            negated_in_sentence.update(neg_terms)

            i = scope_end  # Jump to the end of the current negation scope.

        # 2. Count affirmed mentions in the sentence, excluding terms already negated there.
        affirmed_terms = tp._term_mentions_in_range(tokens, 0, len(tokens))
        for term in affirmed_terms:
            if term not in negated_in_sentence:
                pos_counts[term] += 1

    return dict(neg_counts), dict(pos_counts)



if __name__ == "__main__":
    # Run the corpus-level negation statistics utility.

    main()
