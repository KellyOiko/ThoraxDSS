# evaluation/synonym_corpus_stats.py

import sys
from pathlib import Path
import copy

import pandas as pd
import matplotlib.pyplot as plt

# Resolve project root robustly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.negation import TermConfig, TermProcessor  # noqa: E402


def find_first_existing(candidates):
    for p in candidates:
        p = Path(p)
        if p.exists():
            return p
    return None


def rglob_first(patterns):
    for pat in patterns:
        hits = list(PROJECT_ROOT.rglob(pat))
        if hits:
            hits = sorted(hits, key=lambda x: len(str(x)))
            return hits[0]
    return None


def resolve_reports_csv():
    candidates = [
        PROJECT_ROOT / "backend" / "data" / "processed" / "openi_pairs.csv",
        PROJECT_ROOT / "backend" / "data" / "openi_pairs.csv",
        PROJECT_ROOT / "data" / "openi_pairs.csv",
        PROJECT_ROOT / "backend" / "data" / "processed" / "openi_pairs_clean.csv",
        PROJECT_ROOT / "backend" / "data" / "processed" / "pairs.csv",
    ]
    p = find_first_existing(candidates)
    if p:
        return p
    return rglob_first(["openi_pairs*.csv", "*pairs*.csv"])


def resolve_vocab_xlsx():
    candidates = [
        PROJECT_ROOT / "backend" / "data" / "radiology_vocabulary_final.xlsx",
        PROJECT_ROOT / "backend" / "radiology_vocabulary_final.xlsx",
        PROJECT_ROOT / "data" / "radiology_vocabulary_final.xlsx",
    ]
    p = find_first_existing(candidates)
    if p:
        return p
    return rglob_first(["radiology_vocabulary_final.xlsx", "*vocabulary*.xlsx", "*synonym*.xlsx"])


# Config
TEXT_COL = "text"
REPORTS_CSV = resolve_reports_csv()
VOCAB_XLSX = resolve_vocab_xlsx()
OUT_DIR = PROJECT_ROOT / "Figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


print("=== PATH CHECK ===")
print("PROJECT_ROOT:", PROJECT_ROOT)
print("REPORTS_CSV:", REPORTS_CSV)
print("VOCAB_XLSX :", VOCAB_XLSX)
print("OUT_DIR    :", OUT_DIR)
print("==================\n")

if REPORTS_CSV is None:
    raise FileNotFoundError("Could not find openi_pairs*.csv (update resolve_reports_csv).")
if VOCAB_XLSX is None:
    raise FileNotFoundError("Could not find radiology_vocabulary_final.xlsx (update resolve_vocab_xlsx).")

# Load texts
df = pd.read_csv(REPORTS_CSV)

print("df rows:", len(df))

# Always check column exists BEFORE using it
if TEXT_COL not in df.columns:
    raise ValueError(f"CSV missing column '{TEXT_COL}'. Found: {df.columns.tolist()}")

# Compute diagnostics safely
non_empty_mask = df[TEXT_COL].fillna("").astype(str).str.strip() != ""
print("non-empty text rows:", int(non_empty_mask.sum()))

# Keep only non-empty texts (this is what fixes denominator)
df = df[non_empty_mask].copy()

texts = df[TEXT_COL].fillna("").astype(str).tolist()
print("texts rows (after filtering):", len(texts))
print(f"Loaded {len(texts):,} texts (non-empty only).\n")

# Build processors
# Full: loads terms + synonyms from XLSX
cfg_full = TermConfig(vocab_xlsx_path=str(VOCAB_XLSX))
tp_full = TermProcessor(cfg_full)

# Canonical-only: copy cfg_full so we DON'T re-run TermConfig.__post_init__()
cfg_canon = copy.deepcopy(cfg_full)
cfg_canon.synonyms_map = {}  # remove synonym variants
tp_canon = TermProcessor(cfg_canon)

# Sanity check: phrase variants should be larger when synonyms enabled
print("=== SANITY CHECK ===")
print("n_terms:", len(cfg_full.terms))
print("n_syn_map_keys:", len(cfg_full.synonyms_map))
print("phrase_variants canonical-only:", len(tp_canon.phrase_variants))
print("phrase_variants with-synonyms :", len(tp_full.phrase_variants))
print("====================\n")

if len(tp_full.phrase_variants) == len(tp_canon.phrase_variants):
    print("WARNING: phrase_variants are the same size — synonyms may not be loading as expected.\n")


def matched_canonical_terms(tp: TermProcessor, text: str):
    tokens = tp._tokenize(text)
    found = set()
    for canon_norm, phrase_tokens in tp.phrase_variants:
        if phrase_tokens and tp._find_phrase_starts(tokens, phrase_tokens):
            found.add(canon_norm)
    return found


# Compute synonym gain per report
canon_counts = []
full_counts = []
gains = []

# extra useful stat: how often any SYNONYM surface form fired
reports_with_any_syn_surface = 0

for text in texts:
    canon_set = matched_canonical_terms(tp_canon, text)
    full_set = matched_canonical_terms(tp_full, text)

    c = len(canon_set)
    f = len(full_set)

    canon_counts.append(c)
    full_counts.append(f)
    gains.append(f - c)

    # synonym surface-form detection:
    # a synonym helps if full matched something that canonical didn't
    if f > c:
        reports_with_any_syn_surface += 1


stats_df = pd.DataFrame({
    "canonical_matches": canon_counts,
    "with_synonyms_matches": full_counts,
    "gain": gains
})
stats_path = OUT_DIR / "synonym_gain_summary.csv"
stats_df.to_csv(stats_path, index=False)

# Plot histogram
max_gain = max(gains) if gains else 0
bins = [x - 0.5 for x in range(0, max_gain + 2)]  # integer bins

# Plot histogram of gains with integer bins
import numpy as np

plt.figure()
bins = np.arange(-0.5, max_gain + 1.5, 1)  # bins centered on integers
#plt.hist(gains, bins=bins)
plt.hist(gains, bins=bins, rwidth=0.8)

plt.xlabel("Synonym expansion gain (extra concepts per report)")
plt.ylabel("Number of reports")
plt.xticks(range(0, max_gain + 1))
plt.tight_layout()

fig_path = OUT_DIR / "synonym_gain_per_report_hist.png"
plt.savefig(fig_path, dpi=300)
plt.close()


# Headline stats
gain_nonzero = sum(1 for g in gains if g > 0)

print("Saved:")
print(" -", stats_path)
print(" -", fig_path)
print("\nHeadline stats:")
print(f" - reports with gain > 0: {gain_nonzero:,} / {len(gains):,} ({100.0*gain_nonzero/len(gains):.1f}%)")
print(f" - mean gain: {sum(gains)/len(gains):.3f}")
print(f" - median gain: {pd.Series(gains).median():.0f}")
print(f" - max gain: {max_gain}")
print(f" - reports where synonym expansion helped (f>c): {reports_with_any_syn_surface:,}")
