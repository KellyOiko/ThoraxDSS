"""
Negation-aware term loading, synonym handling, and polarity-based comparison utilities.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Iterable, Tuple
import re
from pathlib import Path
import pandas as pd


# ------------------ helpers ------------------

def normalize_term(t: str) -> str:
    t = (t or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def _resolve_xlsx_path(xlsx_path: str) -> Optional[str]:
    """
    Resolve an XLSX path robustly across absolute and project-relative locations.

    - If the provided path already exists, use it directly.
    - Otherwise, try common locations relative to the backend and project root.
    """
    if not xlsx_path:
        return None

    p = Path(xlsx_path).expanduser()
    if p.is_file():
        return str(p)

    here = Path(__file__).resolve().parent          # backend/
    project_root = here.parent                      # root/

    candidates = [
        here / xlsx_path,                           # backend/<xlsx_path>
        project_root / xlsx_path,                   # root/<xlsx_path>
        here / "data" / xlsx_path,                  # backend/data/<xlsx_path> when only a filename is provided
        project_root / "data" / xlsx_path,          # root/data/<xlsx_path>
    ]
    for c in candidates:
        if c.is_file():
            return str(c)

    return None


def _get_synonym_col_indices(df: pd.DataFrame) -> List[int]:
    """
    Handle Excel files with duplicate 'Synonym' columns.

    When duplicate headers are present, accessing a row by column name may only
    return the first matching column, so synonym columns are tracked by index.
    """
    idxs: List[int] = []
    for i, col in enumerate(df.columns):
        name = str(col).strip().lower()
        if name.startswith("synonym"):
            idxs.append(i)
    return idxs


# ------------------ XLSX loaders ------------------

def load_terms_from_openi_xlsx(xlsx_path: str, sheet_name: str = "synonyms", term_col: str = "Term") -> List[str]:
    """
    Load canonical terms from an OpenI-style Excel sheet.

    Terms are returned as they appear in the file without deduplication or pruning.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    # Normalize column names for more robust matching.
    cols_norm = {str(c).strip().lower(): c for c in df.columns}
    if term_col.lower() not in cols_norm:
        return []

    real_term_col = cols_norm[term_col.lower()]
    out: List[str] = []

    for v in df[real_term_col].tolist():
        if isinstance(v, str) and v.strip():
            out.append(v.strip())

    return out


def load_synonyms_map_from_xlsx(
    xlsx_path: str,
    canonical_terms: Iterable[str],
    sheet_name: str = "synonyms",
    term_col: str = "Term",
    max_synonyms_per_term: int = 50,
) -> Dict[str, List[str]]:
    """
    Returns: { canonical_term_normalized -> [syn1_norm, syn2_norm, ...] }

    - robust to duplicate 'Synonym' columns (uses indices)
    - does not deduplicate synonyms
    """
    if not xlsx_path:
        return {}

    try:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    except Exception:
        return {}

    cols_norm = {str(c).strip().lower(): c for c in df.columns}
    if term_col.lower() not in cols_norm:
        return {}

    real_term_col = cols_norm[term_col.lower()]
    syn_col_idxs = _get_synonym_col_indices(df)
    if not syn_col_idxs:
        return {}

    # Pre-normalize the term column to simplify canonical-term matching.
    df_term_norm = df[real_term_col].astype(str).map(normalize_term)

    out: Dict[str, List[str]] = {}

    for raw_term in canonical_terms:
        canon_norm = normalize_term(raw_term)
        if not canon_norm:
            continue

        matches = df[df_term_norm == canon_norm]
        if matches.empty:
            # Optional fallback: try a contains-based match when exact matching fails.
            mask = df[real_term_col].astype(str).str.contains(canon_norm, case=False, na=False)
            cand = df[mask]
            if len(cand) == 1:
                matches = cand

        if matches.empty:
            continue

        row = matches.iloc[0]

        syns_norm: List[str] = []
        for ci in syn_col_idxs:
            v = row.iloc[ci]
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            v = str(v).strip()
            if not v or v.lower() == "nan":
                continue

            # Split safely in case multiple synonyms appear in the same cell.
            parts = re.split(r"[;,]", v)
            for p in parts:
                s = normalize_term(p)
                if s and s != canon_norm:
                    syns_norm.append(s)

        # Keep the synonym list as loaded, only applying the maximum-length cap.
        out[canon_norm] = syns_norm[:max_synonyms_per_term]

    return out


# ------------------ fallback terms (keep if XLSX missing) ------------------

RAW_TERMS = [
    "pneumothorax",
    "pleural effusion",
    "pneumonia",
    "edema",
    "atelectasis",
    "consolidation",
    "interstitial lung disease",
    "pulmonary edema",
    "lung cancer",
    "cystic fibrosis",
    "sarcoidosis",
    "tuberculosis",
    "asbestosis",
    "hemothorax",
    "pneumomediastinum",
    "mediastinal mass",
    "pericardial effusion",
    "cardiomegaly",
    "pleural thickening",
    "fibrosis",
    "atelectasis of lung base",
    "cavitary lesion",
    "hilar lymphadenopathy",
    "pneumoperitoneum",
    "subcutaneous emphysema",
]

DEFAULT_TERMS = list(RAW_TERMS)


@dataclass
class TermConfig:
    """
    Term configuration with optional XLSX vocabulary loading.

    The main application should not load synonym resources directly;
    that responsibility is handled here.
    """
    terms: List[str] = field(default_factory=list)
    vocab_xlsx_path: Optional[str] = None
    synonyms_map: Dict[str, List[str]] = field(default_factory=dict)

    load_warning: str = ""
    loaded_xlsx_path: str = ""

    # Negation cues / terminators / etc (can be customized if needed)
    negation_cues = [
        ["no"], ["without"], ["free", "of"], ["negative", "for"],
        ["no", "evidence", "of"], ["no", "sign", "of"], ["no", "signs", "of"],
        ["there", "is", "no"], ["there", "are", "no"], ["is", "no"], ["are", "no"],
    ]
    neg_scope_terminators = {".", "\n", ";", "but", "however", "though", "although", "except", "nevertheless", "yet", "–", "—"}
    list_connectors = {",", "and", "or", "/", "-"}
    skip_in_match = {",", ":", ";", "-", "/", "(", ")", "[", "]"}

    def __post_init__(self):
        # Resolve the XLSX path, either from the explicit config value or from known default locations.
        xlsx_path = _resolve_xlsx_path(self.vocab_xlsx_path or "")

        if not xlsx_path:
            # Auto-detect the vocabulary file in common project locations.
            here = Path(__file__).resolve().parent  # backend/
            candidates = [
                here / "data" / "radiology_vocabulary_final.xlsx",
                here / "radiology_vocabulary_final.xlsx",
                here.parent / "data" / "radiology_vocabulary_final.xlsx",
            ]
            for p in candidates:
                if p.is_file():
                    xlsx_path = str(p)
                    break

        # If an XLSX file is available, load canonical terms and their synonym mappings.
        if xlsx_path:
            self.loaded_xlsx_path = xlsx_path
            try:
                xlsx_terms = load_terms_from_openi_xlsx(xlsx_path)
                if xlsx_terms:
                    self.terms = xlsx_terms
                    # Note: Use the exact loaded canonical term list when building the synonym map.
                    self.synonyms_map = load_synonyms_map_from_xlsx(xlsx_path, canonical_terms=self.terms)
                    return
                else:
                    self.load_warning = f"XLSX loaded but Term column produced 0 terms. path={xlsx_path}"
            except Exception as e:
                self.load_warning = f"Failed to load XLSX ({e}). path={xlsx_path}"

        # Fall back to the built-in default term list if no XLSX vocabulary is available.
        if not self.terms:
            self.terms = list(DEFAULT_TERMS)
        if not self.synonyms_map:
            self.synonyms_map = {}
        if not self.load_warning and not self.loaded_xlsx_path:
            self.load_warning = "No XLSX found; using DEFAULT_TERMS."


class TermProcessor:
    """Tokenization and canonical term matching with punctuation skipping and synonym support."""
    
    def __init__(self, config: TermConfig):
        self.config = config

        # Store canonical terms in normalized form.
        self.terms = [normalize_term(t) for t in (config.terms or []) if normalize_term(t)]
        terms_set = set(self.terms)

        # Phrase variants are stored as (canonical_term, tokenized_variant).
        self.phrase_variants: List[Tuple[str, List[str]]] = []

        # Add canonical term variants.
        for t in self.terms:
            toks = t.split()
            if toks:
                self.phrase_variants.append((t, toks))

        # Map synonym variants back to their canonical term.
        for canon_norm, syn_list in (config.synonyms_map or {}).items():
            canon_norm = normalize_term(canon_norm)
            if canon_norm not in terms_set:
                continue
            for s in (syn_list or []):
                syn_norm = normalize_term(s)
                if not syn_norm or syn_norm == canon_norm:
                    continue
                toks = syn_norm.split()
                if toks:
                    self.phrase_variants.append((canon_norm, toks))

        self.token_re = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")




    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words, numbers, and punctuation markers."""

        text = (text or "").lower()
        text = re.sub(r"\s+", " ", text).strip()
        return self.token_re.findall(text)


    def _split_sentences(self, text: str) -> List[str]:
        """Split text into coarse sentence-like units for negation scope handling."""

        parts = re.split(r"[.\n;]+", (text or ""))
        return [p.strip() for p in parts if p.strip()]


    def _find_phrase_starts(self, tokens: List[str], phrase_tokens: List[str]) -> List[int]:
        """Find token positions where a phrase appears, allowing punctuation skipping."""
        
        if not phrase_tokens:
            return []
        starts = []
        L, P = len(tokens), len(phrase_tokens)

        for start in range(L):
            ti, pi = start, 0
            while ti < L and pi < P:
                if tokens[ti] in self.config.skip_in_match:
                    ti += 1
                    continue
                if tokens[ti] == phrase_tokens[pi]:
                    ti += 1
                    pi += 1
                else:
                    break
            if pi == P:
                starts.append(start)
        return starts

    def _term_mentions_in_range(self, tokens: List[str], lo: int, hi: int) -> Set[str]:
        """Return canonical terms mentioned within a token range."""
       
        window = tokens[lo:hi]
        found: Set[str] = set()

        # Match any known variant, but always record the canonical term.
        for canon, phrase_tokens in self.phrase_variants:
            if phrase_tokens and self._find_phrase_starts(window, phrase_tokens):
                found.add(canon)

        return found

    def _match_phrase_at(self, tokens: List[str], start: int, phrase: List[str]) -> bool:
        """Check whether a token sequence matches a phrase exactly at a given position."""
        
        if start + len(phrase) > len(tokens):
            return False
        return tokens[start:start + len(phrase)] == phrase

    def build_polarity_map(self, text: str) -> Dict[str, int]:
        """
        term -> polarity:
          +1 = affirmed/present
          -1 = negated
           0 = not mentioned OR conflict
        """
        pols: Dict[str, Set[int]] = {t: set() for t in self.terms}

        for sent in self._split_sentences(text):
            tokens = self._tokenize(sent)
            if not tokens:
                continue

            negated_in_sentence: Set[str] = set()

            # 1) Mark terms in negation scope as -1
            i = 0
            while i < len(tokens):
                cue = None
                for c in self.config.negation_cues:
                    if self._match_phrase_at(tokens, i, c):
                        cue = c
                        break

                if cue is None:
                    i += 1
                    continue

                scope_start = i + len(cue)

                while scope_start < len(tokens) and tokens[scope_start] in self.config.list_connectors:
                    scope_start += 1

                scope_end = len(tokens)
                for j in range(scope_start, len(tokens)):
                    if tokens[j] in self.config.neg_scope_terminators:
                        scope_end = j
                        break

                neg_terms = self._term_mentions_in_range(tokens, scope_start, scope_end)
                for term in neg_terms:
                    pols[term].add(-1)
                negated_in_sentence.update(neg_terms)

                i = scope_end

            # 2) Add +1 for affirmed mentions NOT negated in this sentence
            affirmed_terms = self._term_mentions_in_range(tokens, 0, len(tokens))
            for term in affirmed_terms:
                if term not in negated_in_sentence:
                    pols[term].add(+1)

        # Resolve conflicts
        result: Dict[str, int] = {}
        for term, polarities in pols.items():
            if +1 in polarities and -1 in polarities:
                result[term] = 0  # Conflicting evidence -> unknown / ambiguous.
            elif -1 in polarities:
                result[term] = -1
            elif +1 in polarities:
                result[term] = +1
            else:
                result[term] = 0

        return result



class NegationAdjuster:
    """Compute negation-aware comparison signals and retrieval score adjustments."""

    def __init__(self, term_processor: TermProcessor):
        self.term_processor = term_processor

        # Base contributions
        self.bonus_pos = 0.35     #  {term present in both (+1,+1)}
        self.bonus_neg = 0.15     #  {term absent in both (-1,-1)} 
        
        # Strong penalty when polarity conflicts (+1 vs -1)
        self.penalty_mismatch = 0.7

        # Mild penalty when a term appears in only one text.
        # This helps capture mention mismatch even without direct polarity conflict.
        self.penalty_missing = 0.20

        # Scale the normalized adjustment before clamping.
        # This increases the overall influence of negation-aware scoring.
        self.scale = 1.2

    def _pair_base(self, p: int, r: int) -> float:
        # both not mentioned -> no effect
        if p == 0 and r == 0:
            return 0.0

        # one mentioned, the other not -> mild negative signal
        if (p == 0) != (r == 0):
            return -self.penalty_missing

        # both mentioned
        if p == r == +1:
            return self.bonus_pos
        if p == r == -1:
            return self.bonus_neg

        # polarity conflict
        return -self.penalty_mismatch

    def compute_adjustment(
        self,
        patient_text: str,
        report_text: str,
        max_abs: float = 0.35,           # Upper bound on the final adjustment magnitude.
        normalize_by_terms: bool = True
    ) -> float:
        pm = self.term_processor.build_polarity_map(patient_text)
        rm = self.term_processor.build_polarity_map(report_text)

        adj = 0.0
        used = 0

        for term in self.term_processor.terms:
            p = pm.get(term, 0)
            r = rm.get(term, 0)

            # count only terms that are mentioned in at least one text
            if p != 0 or r != 0:
                used += 1
                adj += self._pair_base(p, r)

        if normalize_by_terms:
            denom = used if used > 0 else 1
            adj = adj / denom

        # Apply global scaling before clamping.
        adj = adj * self.scale

        # clamp
        if adj > max_abs:
            return max_abs
        if adj < -max_abs:
            return -max_abs
        return adj

    def get_negation_warnings(self, patient_text: str, report_text: str) -> List[str]:
        pm = self.term_processor.build_polarity_map(patient_text)
        rm = self.term_processor.build_polarity_map(report_text)

        warnings = []
        for term in self.term_processor.terms:
            p = pm.get(term, 0)
            r = rm.get(term, 0)
            if p in (+1, -1) and r in (+1, -1) and p != r:
                p_str = "present" if p == +1 else "absent"
                r_str = "present" if r == +1 else "absent"
                warnings.append(f'Negation mismatch: "{term}" patient={p_str} vs report={r_str}')
        return warnings

    def compare_findings(self, patient_text: str, report_text: str) -> Dict[str, List[str]]:
        pm = self.term_processor.build_polarity_map(patient_text)
        rm = self.term_processor.build_polarity_map(report_text)

        common, patient_only, report_only = [], [], []

        def fmt(term: str, pol: int) -> str:
            if pol == +1:
                return f"{term} [present]"
            if pol == -1:
                return f"{term} [absent]"
            return f"{term} [unknown]"

        for term in self.term_processor.terms:
            p = pm.get(term, 0)
            r = rm.get(term, 0)

            if p in (-1, +1) and r in (-1, +1) and p == r:
                common.append(fmt(term, p))
                continue
            if p in (-1, +1) and not (r in (-1, +1) and r == p):
                patient_only.append(fmt(term, p))
            if r in (-1, +1) and not (p in (-1, +1) and p == r):
                report_only.append(fmt(term, r))

        return {"common": common, "patient_only": patient_only, "report_only": report_only}


if __name__ == "__main__":
    # Demo configuration using the existing term-processing pipeline.

    cfg = TermConfig(vocab_xlsx_path="backend/data/radiology_vocabulary_final.xlsx")
    tp = TermProcessor(cfg)
    na = NegationAdjuster(tp)

    print("\n=== TERM RESOURCE LOAD ===")
    print("loaded_xlsx_path:", cfg.loaded_xlsx_path)
    print("load_warning:", cfg.load_warning)
    print("n_terms:", len(cfg.terms))
    print("n_synonyms_map_keys:", len(cfg.synonyms_map))

    # Example of manually selecting a canonical term for inspection:
    # example_term = "atelectasis"
    # ex_norm = normalize_term(example_term)

    print("\n=== EXAMPLE TERM + SYNONYMS (if available) ===")
    # cfg.synonyms_map is keyed by normalized canonical terms.
    # syns = cfg.synonyms_map.get(ex_norm, [])


    # Pick one canonical term that actually has loaded synonyms.
    example_key = next((k for k, v in cfg.synonyms_map.items() if v), None)


    if example_key is None:
        print("No synonyms available in cfg.synonyms_map.")
    else:
        syns = cfg.synonyms_map[example_key]
        print(f"canonical term (normalized key): {example_key}")
        print(f"synonyms loaded (first 15): {syns[:15]}{' ...' if len(syns) > 15 else ''}")


    # --- Two short texts crafted to exercise synonym handling and negation scope ---
    patient_text = (
        "No evidence of pneumonitis or pneumothorax is seen. "
    )
    report_text = (
        "Pneumonia is identified. "
        "Free air in the chest outside the lung."
    )


    print("\n=== INPUT TEXTS ===")
    print("PATIENT:", patient_text)
    print("REPORT: ", report_text)

    # --- Show polarity maps (term -> polarity:+1 / -1 / 0) ---
    pm = tp.build_polarity_map(patient_text)
    rm = tp.build_polarity_map(report_text)

    # Print only terms that are mentioned in either text (so output stays readable)
    def mentioned_only(m):
        return {k: v for k, v in m.items() if v != 0}

    print("\n=== POLARITY MAPS (mentioned terms only) ===")
    print("PATIENT polarity:", mentioned_only(pm))
    print("REPORT polarity: ", mentioned_only(rm))

    # --- Show warnings and structured overlap/differences ---
    print("\n=== NEGATION WARNINGS ===")
    print(na.get_negation_warnings(patient_text, report_text))

    print("\n=== COMPARE FINDINGS (canonical terms + polarity tags) ===")
    print(na.compare_findings(patient_text, report_text))

    # --- Show scalar adjustment used in retrieval ---
    print("\n=== NEGATION-AWARE ADJUSTMENT ===")
    print("compute_adjustment:", na.compute_adjustment(patient_text, report_text))
