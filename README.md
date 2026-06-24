# 🩻 ThoraxDSS — Multimodal RAG for Chest X-ray Analysis

> **Diploma Thesis** • Department of Computer Science and Engineering, University of Ioannina (CSE UOI)

ThoraxDSS is a multimodal clinical decision support system for chest X-ray interpretation.  
It combines multimodal retrieval, negation-aware radiology term processing, and vision-language models to produce structured, evidence-grounded analyses.

## 🧠 Key Features

- Multimodal retrieval with FAISS
- Fusion of retrieval pathways with configurable weighting
- Negation-aware and synonym-aware matching of radiological findings
- Structured JSON generation with vision-language models
- Interactive Streamlit-based user interface
- Evaluation utilities for retrieval-weight sweeps and multi-patient experiments

---

## 🚀 System Functionality

ThoraxDSS accepts:
- a clinical summary
- a chest X-ray image

It then performs multimodal similarity search across three parallel retrieval pathways:
- Text→Text (T2T)
- Text→Image (T2I)
- Image→Image (I2I)

The retrieved signals are fused through configurable pathway weights and subsequently passed to a vision-language model for structured analysis generation.

The system generates:
- patient findings
- retrieved-case comparisons
- final diagnosis assessment

---

## Why Negation Matters

Radiology reports routinely describe both affirmed and negated findings.
ThoraxDSS incorporates negation-aware processing to prevent clinically opposite statements from being treated as equivalent during retrieval, comparison, and final evidence synthesis.

---

## ⚙️ Requirements

- Python 3.10+
- PyTorch (with CPU or GPU support)
- Ollama or a compatible vision-language model serving endpoint

The project dependencies are listed in `requirements.txt` and include:

- Core libraries: PyTorch, Transformers, Sentence-Transformers, OpenCLIP, FAISS
- Data handling: NumPy, Pandas, PyArrow / FastParquet
- Interface: Streamlit
- Evaluation: BLEU, ROUGE, BERTScore, and related utilities

Install all dependencies with:
```bash
pip install -r requirements.txt 
```
---

## 🖥️ How to Run

### Launch the Application
```bash
streamlit run frontend/app.py
```

---

## 📌 Notes

- This repository does not include medical images, radiology reports, or precomputed experimental artifacts.
- The experimental pipeline is based on the Open-I chest X-ray dataset, which is publicly available but subject to data usage and redistribution conditions; therefore, the original data are not included in this repository.
- Data files and model artifacts must be prepared separately.
- This project was developed for research and thesis purposes and is not intended for clinical use.


## 📄 Thesis

The accompanying thesis is available [here](multimodal_cxr_rag_ui_public/thesis/thesis.pdf).
