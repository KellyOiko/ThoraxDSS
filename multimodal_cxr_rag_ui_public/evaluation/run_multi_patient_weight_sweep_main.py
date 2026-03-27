"""
Entry script for running multi-patient retrieval-weight sweeps.

Forces execution under an ASCII project path to avoid FAISS issues on Windows,
then loads data and launches the sweep.
"""
from pathlib import Path
import sys
import os
import pandas as pd

# Force ASCII project root (FAISS cannot reliably read Unicode paths on Windows)
PREFERRED_ROOT = Path(r"C:\mm_cxr")  # ASCII-safe path
if not (PREFERRED_ROOT / "backend" / "main.py").exists():
    raise RuntimeError(
        f"Expected ASCII clone at {PREFERRED_ROOT} but backend/main.py not found.\n"
        f"Either clone/copy your project to C:\\mm_cxr, or change PREFERRED_ROOT."
    )

PROJECT_ROOT = PREFERRED_ROOT

# Ensure imports and execution happen from the ASCII project root
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

print("[BOOT] PROJECT_ROOT =", PROJECT_ROOT)
print("[BOOT] CWD          =", Path.cwd())

# Imports AFTER sys.path / cwd fix
from evaluation.run_multi_patient_weight_sweep import run_multi_patient_weight_sweep

# Paths aligned with backend.config.DataConfig (under ASCII root)
DF_SAMPLE_PATH = PROJECT_ROOT / "data" / "processed" / "df_sample.parquet"
IMAGE_ROOT     = PROJECT_ROOT / "data" / "raw" / "NLMCXR_png"
OUT_DIR        = PROJECT_ROOT / "evaluation" / "_multi_patient_sweep"

# Sanity checks before FAISS is accessed
print("[PATH] df_sample:", DF_SAMPLE_PATH, "exists:", DF_SAMPLE_PATH.exists())
print("[PATH] images   :", IMAGE_ROOT, "exists:", IMAGE_ROOT.exists())
print("[PATH] faiss txt:", (PROJECT_ROOT / "data/processed/faiss_text.index"), "exists:",
      (PROJECT_ROOT / "data/processed/faiss_text.index").exists())
print("[PATH] faiss img:", (PROJECT_ROOT / "data/processed/faiss_image.index"), "exists:",
      (PROJECT_ROOT / "data/processed/faiss_image.index").exists())

if not DF_SAMPLE_PATH.exists():
    raise FileNotFoundError(f"Missing {DF_SAMPLE_PATH}")

if not (PROJECT_ROOT / "data/processed/faiss_text.index").exists():
    raise FileNotFoundError("Missing faiss_text.index under ASCII root data/processed")
if not (PROJECT_ROOT / "data/processed/faiss_image.index").exists():
    raise FileNotFoundError("Missing faiss_image.index under ASCII root data/processed")

# Load df_sample (must match embeddings / FAISS subset)
df_sample = pd.read_parquet(DF_SAMPLE_PATH)

# Run the multi-patient sweep
run_multi_patient_weight_sweep(
    df_sample=df_sample,
    image_root=IMAGE_ROOT,
    out_dir=OUT_DIR,
    k_cases=3,
    model="llava",
)
