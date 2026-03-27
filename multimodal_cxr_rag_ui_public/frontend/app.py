"""
Streamlit frontend for the ThoraxDSS multimodal chest X-ray analysis system.

This UI collects the clinical summary and patient image, runs the backend
multimodal RAG pipeline, and renders structured results, retrieval debug data,
and raw model output.
"""
import streamlit as st
from PIL import Image
import os
import tempfile
import json
import sys
from pathlib import Path
import re
import html


# Add the project root to the Python path so the frontend can import backend modules.
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.main import get_system

# ====== Page setup and global UI styling ======
st.set_page_config(page_title="ThoraxDSS", layout="wide")

st.markdown(
    """
    <style>
    /* =========================
       THESIS SCREENSHOT MODE
       ========================= */

    /* Tighten layout so big fonts don't waste space */
    [data-testid="stAppViewContainer"] .main .block-container{
      padding-top: 1rem !important;
      padding-bottom: 1rem !important;
      max-width: 1100px !important;
    }

    /* -------- Header typography (our custom classes) -------- */
    .thesis-title{
      font-size: 64px !important;
      font-weight: 900 !important;
      line-height: 1.05 !important;
      margin: 0 !important;
    }
    .thesis-subtitle{
      font-size: 34px !important;
      font-weight: 800 !important;
      margin-top: 10px !important;
      line-height: 1.2 !important;
    }
    .thesis-lead{
      font-size: 28px !important;
      margin-top: 14px !important;
      line-height: 1.45 !important;
      color: rgba(255,255,255,0.92) !important;
    }
    .thesis-callout{
      margin-top: 14px !important;
      padding-left: 14px !important;
      border-left: 6px solid #60A5FA !important;
    }
    .thesis-callout b{
      color:#60A5FA !important;
      font-weight: 900 !important;
    }

    /* -------- Form labels (our custom class) -------- */
    .thesis-label{
      font-size: 30px !important;
      font-weight: 800 !important;
      margin: 12px 0 6px 0 !important;
      line-height: 1.2 !important;
    }

    /* -------- Widgets: make them REALLY big -------- */
    /* Text area */
    [data-testid="stTextArea"] textarea{
      font-size: 30px !important;
      line-height: 1.5 !important;
      padding: 18px !important;
    }

    /* Text input */
    [data-testid="stTextInput"] input{
      font-size: 28px !important;
      padding: 14px 16px !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] *{
      font-size: 24px !important;
      line-height: 1.3 !important;
    }

    /* Sliders */
    [data-testid="stSlider"] *{
      font-size: 24px !important;
    }

    /* Expanders */
    [data-testid="stExpander"] summary *{
      font-size: 26px !important;
      font-weight: 800 !important;
    }

    /* Tabs */
    [data-testid="stTabs"] *{
      font-size: 22px !important;
    }

    /* Primary button */
    [data-testid="stButton"] > button{
      font-size: 28px !important;
      padding: 1.1rem 1.2rem !important;
      border-radius: 14px !important;
    }

    /* Captions */
    .stCaption, [data-testid="stCaptionContainer"] *{
      font-size: 20px !important;
    }
    /* ===== Expanders: "Advanced model settings", "Retrieval weights" ===== */
    [data-testid="stExpander"] summary{
    padding: 14px 12px !important;
    }
    [data-testid="stExpander"] summary div,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span{
    font-size: 30px !important;
    font-weight: 900 !important;
    }

    /* ===== Caption inside expander: "Weights must sum to 1..." ===== */
    [data-testid="stCaptionContainer"] *{
    font-size: 24px !important;
    line-height: 1.35 !important;
    }

    /* ===== Slider labels: "Text→Image weight", "Image→Image weight" ===== */
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] label p,
    [data-testid="stSlider"] label span{
    font-size: 28px !important;
    font-weight: 800 !important;
    }

    /* ===== Slider tick labels / min-max values (0.00, 1.00 etc.) ===== */
    [data-testid="stSlider"] [data-testid="stTickBar"] *,
    [data-testid="stSlider"] [data-testid="stTickBarMin"],
    [data-testid="stSlider"] [data-testid="stTickBarMax"],
    [data-testid="stSlider"] [data-testid="stTickBar"] span{
    font-size: 22px !important;
    }

    /* ===== Slider current value bubble ===== */
    [data-testid="stSlider"] [data-testid="stThumbValue"]{
    font-size: 22px !important;
    font-weight: 800 !important;
    }

    /* ===== Markdown line: "Auto Text→Text weight: ..." ===== */
    [data-testid="stMarkdownContainer"] p{
    font-size: 24px !important;
    line-height: 1.35 !important;
    }

    /* Fallback expander header selector (Streamlit versions differ) */
    .streamlit-expanderHeader, .streamlit-expanderHeader *{
    font-size: 30px !important;
    font-weight: 900 !important;
    }
    /* ===== Expander body text (anything inside Advanced / Weights panels) ===== */
    [data-testid="stExpander"] .stMarkdown,
    [data-testid="stExpander"] .stMarkdown p,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p{
    font-size: 26px !important;
    line-height: 1.35 !important;
    }

    /* ===== "Vision-Language Model (served via Ollama):" label (your markdown) ===== */
    [data-testid="stExpander"] p{
    font-size: 28px !important;
    font-weight: 800 !important;
    }


    /* ===== Text input placeholder + text (custom model name) ===== */
    [data-testid="stTextInput"] input{
    font-size: 28px !important;
    padding: 14px 16px !important;
    }

    /* ===== Slider help text + the auto-weight line inside Retrieval weights ===== */
    [data-testid="stExpander"] [data-testid="stCaptionContainer"] *{
    font-size: 24px !important;
    }

    /* =========================
    SELECTBOX — NUCLEAR OVERRIDE (value + dropdown)
    Put LAST in the FIRST <style> block
    ========================= */

    /* Selected value inside the closed control */
    [data-testid="stSelectbox"] div[data-baseweb="select"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stSelectbox"] div[data-baseweb="select"] div,
    [data-testid="stSelectbox"] div[role="combobox"] input,
    [data-testid="stSelectbox"] div[role="combobox"] span,
    [data-testid="stSelectbox"] div[role="combobox"] div{
    font-size: 28px !important;
    font-weight: 900 !important;
    line-height: 1.25 !important;
    opacity: 1 !important;
    visibility: visible !important;
    }

    /* Make the whole selectbox taller so big text doesn't clip */
    [data-testid="stSelectbox"] div[data-baseweb="select"]{
    min-height: 110px !important;
    }
    [data-testid="stSelectbox"] div[role="combobox"]{
    min-height: 82px !important;
    padding: 12px 14px !important;
    }

    /* Force visible text color (dark + light) */
    [data-testid="stSelectbox"] div[data-baseweb="select"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stSelectbox"] div[role="combobox"] input,
    [data-testid="stSelectbox"] div[role="combobox"] span{
    color: rgba(255,255,255,0.92) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.92) !important;
    }
    @media (prefers-color-scheme: light){
    [data-testid="stSelectbox"] div[data-baseweb="select"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stSelectbox"] div[role="combobox"] input,
    [data-testid="stSelectbox"] div[role="combobox"] span{
        color: rgba(0,0,0,0.88) !important;
        -webkit-text-fill-color: rgba(0,0,0,0.88) !important;
    }
    }

    /* Dropdown (opened menu) */
    div[data-baseweb="popover"] *,
    ul[role="listbox"] *,
    li[role="option"] *{
    font-size: 28px !important;
    font-weight: 800 !important;
    line-height: 1.35 !important;
    }
    li[role="option"]{
    padding-top: 14px !important;
    padding-bottom: 14px !important;
    }
    /* =========================
    SELECTBOX — FIX CLIPPING (PUT LAST)
    ========================= */

    /* Make the whole select control use flex so text is vertically centered */
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div{
    display: flex !important;
    align-items: center !important;
    }


    /* Make sure the actual text element doesn't get cut */
    [data-testid="stSelectbox"] div[data-baseweb="select"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] span,
    [data-testid="stSelectbox"] div[role="combobox"] input,
    [data-testid="stSelectbox"] div[role="combobox"] span{
    line-height: 1.35 !important;      /* <-- IMPORTANT (was 1.25) */
    height: auto !important;
    padding: 0 !important;
    margin: 0 !important;
    }

    /* Prevent hidden overflow clipping the text */
    [data-testid="stSelectbox"] div[data-baseweb="select"]{
    overflow: visible !important;
    }

    /* =========================
   RETRIEVAL WEIGHTS — MORE SPACING
   Put LAST in the FIRST <style> block
   ========================= */

    /* (1) Το κείμενο "Weights must sum to 1..." να έχει κενό από κάτω */
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p{
    margin-top: 10px !important;
    margin-bottom: 20px !important;
    }

    /* (2) Κενό πάνω/κάτω από κάθε slider block */
    [data-testid="stExpander"] [data-testid="stSlider"]{
    margin-top: 18px !important;
    margin-bottom: 22px !important;
    }

    /* (3) Να κατεβαίνει λίγο η μπάρα (slider track) κάτω από το label */
    [data-testid="stSlider"] > div{
    padding-top: 22px !important;   /* αύξησε αν ακόμα "μπλέκεται" */
    }

    /* (4) Extra κενό κάτω από το label ("Text→Image weight", "Image→Image weight") */
    [data-testid="stSlider"] label{
    margin-bottom: 14px !important;
    }

    /* (5) Το "Auto Text→Text weight..." να απέχει από τους sliders */
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p strong{
    display: inline-block !important;
    margin-top: 14px !important;
    }

    /* =========================
   BIG REPORT TYPOGRAPHY (PUT LAST)
   ========================= */

    /* Big titles like "Patient Image", "Patient findings", "Retrieved cases", "Final diagnosis assessment" */
    [data-testid="stMarkdownContainer"] h1{ font-size: 56px !important; line-height: 1.1 !important; }
    [data-testid="stMarkdownContainer"] h2{ font-size: 48px !important; line-height: 1.12 !important; }
    [data-testid="stMarkdownContainer"] h3{ font-size: 40px !important; line-height: 1.15 !important; }

    /* General report text, bullets, etc. */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li{
    font-size: 30px !important;
    line-height: 1.35 !important;
    }

    /* Tabs: Analysis / Retrieval debug / Raw output */
    [data-testid="stTabs"] button p,
    [data-testid="stTabs"] button div{
    font-size: 30px !important;
    font-weight: 900 !important;
    }

    /* Alerts: "Structured output has warnings", errors, infos */
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] div{
    font-size: 28px !important;
    line-height: 1.35 !important;
    }

    /* Expander titles inside results ("Show warnings", "Case report") */
    [data-testid="stExpander"] summary *{
    font-size: 30px !important;
    font-weight: 900 !important;
    }

    /* Code blocks (Raw output JSON) */
    [data-testid="stCodeBlock"] *{
    font-size: 22px !important;
    line-height: 1.35 !important;
    }


    </style>
    """,
    unsafe_allow_html=True
)

# Additional styling for case badges, confidence pills, and result sections.
st.markdown(
    """
    <style>
      /* ===== BIG CASE-CARD LABELS + BADGES ===== */

      .case-section-title{
        font-size: 36px !important;
        font-weight: 900 !important;
        line-height: 1.15 !important;
        margin: 14px 0 10px 0 !important;
      }

      .case-badge{
        display: inline-block;
        font-size: 34px !important;
        font-weight: 900 !important;
        padding: 10px 14px !important;
        border-radius: 14px !important;
        border: 2px solid rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.08);
        margin-top: 10px !important;
      }

      .badge-similar{ background: rgba(25,135,84,0.18); border-color: rgba(25,135,84,0.35); }
      .badge-partial{ background: rgba(255,193,7,0.18); border-color: rgba(255,193,7,0.35); }
      .badge-not{    background: rgba(220,53,69,0.18); border-color: rgba(220,53,69,0.35); }

      .final-conf-row{
        font-size: 40px !important;
        font-weight: 900 !important;
        line-height: 1.1 !important;
        margin: 10px 0 12px 0 !important;
      }

      .conf-pill{
        display:inline-block;
        font-size: 38px !important;
        font-weight: 900 !important;
        padding: 8px 14px !important;
        border-radius: 14px !important;
        border: 2px solid rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.08);
        vertical-align: middle;
      }

      .conf-high{   background: rgba(25,135,84,0.18); border-color: rgba(25,135,84,0.35); }
      .conf-medium{ background: rgba(255,193,7,0.18); border-color: rgba(255,193,7,0.35); }
      .conf-low{    background: rgba(220,53,69,0.18); border-color: rgba(220,53,69,0.35); }
    </style>
    """,
    unsafe_allow_html=True
)


# Render the main page header, subtitle, and introductory description.
st.markdown(
    """
    <div style="margin-top: 4px;">
      <div class="thesis-title">🩻 ThoraxDSS</div>

      <div class="thesis-subtitle">
        Clinical decision support for chest X-ray interpretation
      </div>

      <div class="thesis-lead thesis-callout">
        <b>Multimodal RAG (powered by VLMs):</b>
        Retrieves similar prior cases and generates an evidence-grounded structured analysis.
      </div>

      <div class="thesis-lead">
        Provide a clinical summary, upload a chest X-ray, and receive a structured analysis supported by retrieved cases.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


st.markdown("---")


st.markdown(
    """
    <style>
      :root{
        --chip-bg: rgba(0,0,0,0.04);
        --chip-border: rgba(0,0,0,0.08);
        --chip-text: rgba(0,0,0,0.78);

        --finding-text: rgba(0,0,0,0.85);
        --bullet: rgba(0,0,0,0.45);
        --muted: rgba(0,0,0,0.55);
      }

      @media (prefers-color-scheme: dark){
        :root{
          --chip-bg: rgba(255,255,255,0.06);
          --chip-border: rgba(255,255,255,0.12);
          --chip-text: rgba(255,255,255,0.82);

          --finding-text: rgba(255,255,255,0.90);
          --bullet: rgba(255,255,255,0.55);
          --muted: rgba(255,255,255,0.68);
        }
      }

      .tagline { display:flex; align-items:center; gap:8px; margin:4px 0; }
      .chips { display:flex; flex-wrap:wrap; gap:6px; }
        .chip {
        font-size: 26px;
        padding: 6px 14px;
        line-height: 34px;
        }

        .finding-text {
        font-size: 32px;
        line-height: 1.35;
        }

        .muted {
        font-size: 24px;
        line-height: 1.35;
        }

        .tagline{
        gap: 14px;
        margin: 10px 0;
        }

        .bullet{
        font-size: 34px;
        }


      .chip-strong { background: rgba(0, 120, 212, 0.10); border-color: rgba(0,120,212,0.25); }
      .chip-warn   { background: rgba(255, 170, 0, 0.12); border-color: rgba(255,170,0,0.28); }
      .chip-bad    { background: rgba(220, 53, 69, 0.10); border-color: rgba(220,53,69,0.22); }
      .chip-ok     { background: rgba(25, 135, 84, 0.10); border-color: rgba(25,135,84,0.22); }

        .finding-text { color: var(--finding-text); }
        .bullet { margin-right: 6px; color: var(--bullet); }
        .muted { color: var(--muted); }


    </style>
    """,
    unsafe_allow_html=True
)


# ====== Main input form ======
st.markdown("<div class='thesis-label'>Clinical summary / doctor's findings:</div>", unsafe_allow_html=True)
query_text = st.text_area(
    label="",
    height=300,  # important: big font needs less "empty" height
    placeholder="e.g. Large right pneumothorax with mediastinal shift...",
    label_visibility="collapsed",
)

st.markdown("<div class='thesis-label'>Upload patient chest X-ray:</div>", unsafe_allow_html=True)
uploaded_img = st.file_uploader(
    label="",
    type=["png", "jpg", "jpeg"],
    label_visibility="collapsed",
)


st.markdown("<div class='thesis-label'>Number of retrieved similar cases (k):</div>", unsafe_allow_html=True)
k = st.slider(
    label="",
    min_value=1,
    max_value=10,
    value=3,
    label_visibility="collapsed",
)


# ====== Advanced model options ======
with st.expander("Advanced model settings"):
    st.markdown("<div class='thesis-label'>Vision-Language Model (served via Ollama):</div>", unsafe_allow_html=True)


    model_choice = st.selectbox(
        label="",
        options=["LLaVA (default)", "MiniCPM-V", "Custom (type name)"],
        index=0,
        label_visibility="collapsed",
    )

    custom_model_name = ""
    if model_choice == "Custom (type name)":
        custom_model_name = st.text_input(
            label="",
            placeholder="e.g. phi3, qwen2-vl, internvl...",
            label_visibility="collapsed",
        )



if model_choice == "MiniCPM-V":
    model_name = "minicpm-v"
elif model_choice == "LLaVA (default)":
    model_name = "llava"
else:
    model_name = custom_model_name.strip() or "llava" # Fall back to llava if no custom model name is provided.

# ====== Weights configuration ======
with st.expander("Retrieval weights"):
    st.markdown("<div style='font-size:26px; font-weight:700;'>Weights must sum to 1. Text→Text is computed automatically.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        weight_text2image = st.slider(
            "Text→Image weight",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            key="w_t2i",
        )

    with col2:
        weight_image2image = st.slider(
            "Image→Image weight",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            key="w_i2i",
        )

    # Keep the weights valid and reserve some weight mass for text-to-text retrieval.
    w_t2i = float(weight_text2image)
    w_i2i = float(weight_image2image)

    # If Text→Image and Image→Image become too large together, scale them down
    # so Text→Text keeps at least a small positive weight.
    if w_t2i + w_i2i > 0.95:
        scale = 0.95 / (w_t2i + w_i2i)
        w_t2i *= scale
        w_i2i *= scale
        st.warning("Scaled Text→Image and Image→Image so Text→Text stays ≥ 0.05.")

    weight_text2text = max(0.0, 1.0 - w_t2i - w_i2i)

    st.markdown(
        f"**Auto Text→Text weight:** `{weight_text2text:.2f}`  "
        f"(effective: t2i={w_t2i:.2f}, i2i={w_i2i:.2f})"
    )



# ====== UI Helpers ======

def render_badge(text: str):
    t = (text or "").strip().lower()
    if t == "similar":
        return '<span class="case-badge badge-similar">🟢 SIMILAR</span>'
    if t == "partially similar":
        return '<span class="case-badge badge-partial">🟡 PARTIALLY SIMILAR</span>'
    if t == "not similar":
        return '<span class="case-badge badge-not">🔴 NOT SIMILAR</span>'
    safe = html.escape(text or "")
    return f'<span class="case-badge">{safe}</span>'


def render_confidence(conf: str):
    c = (conf or "").strip().lower()
    if c == "high":
        return '<span class="conf-pill conf-high">🟢 HIGH</span>'
    if c == "medium":
        return '<span class="conf-pill conf-medium">🟡 MEDIUM</span>'
    if c == "low":
        return '<span class="conf-pill conf-low">🔴 LOW</span>'
    safe = html.escape(conf or "")
    return f'<span class="conf-pill">{safe}</span>'

def safe_list(x):
    """Return the input if it is a list, otherwise return an empty list."""
    return x if isinstance(x, list) else []


def safe_dict(x):
    """Return the input if it is a dict, otherwise return an empty dict."""
    return x if isinstance(x, dict) else {}

# Regex and helpers for parsing tagged finding strings.
TAG_PATTERN = re.compile(
    r'^\s*\[LAT=(?P<lat>right|left|bilateral|unknown)\]\s*'
    r'\[SIZE=(?P<size>small|moderate|large|unknown)\]\s*'
    r'\[SHIFT=(?P<shift>yes|no|unknown)\]\s*'
    r'\[DIR=(?P<dir>left|right|none|unknown)\]\s*'
    r'(?P<text>.+)$',
    re.IGNORECASE
)

def parse_tagged_line(s: str) -> tuple[dict, str]:
    """
    Parse a tagged finding line of the form:

    "[LAT=..] [SIZE=..] [SHIFT=..] [DIR=..] finding text"

    Returns:
    - a tag dictionary
    - the finding text with tags removed

    If parsing fails, returns ({}, original_string).
    """
    if not isinstance(s, str):
        return {}, ""
    
    # Try strict pattern first
    m = TAG_PATTERN.match(s.strip())
    if m:
        tags = {
            "lat": (m.group("lat") or "").strip().lower(),
            "size": (m.group("size") or "").strip().lower(),
            "shift": (m.group("shift") or "").strip().lower(),
            "dir": (m.group("dir") or "").strip().lower(),
        }
        text = (m.group("text") or "").strip()
        return tags, text
    
    # Fallback: Try to extract any tags that exist
    tags = {}
    text = s
    
    # Extract LAT if present
    lat_match = re.search(r'\[LAT=([^\]]+)\]', s)
    if lat_match:
        tags["lat"] = lat_match.group(1).strip().lower()
        text = text.replace(lat_match.group(0), "").strip()
    
    # Extract SIZE if present
    size_match = re.search(r'\[SIZE=([^\]]+)\]', s)
    if size_match:
        tags["size"] = size_match.group(1).strip().lower()
        text = text.replace(size_match.group(0), "").strip()
    
    # Extract SHIFT if present
    shift_match = re.search(r'\[SHIFT=([^\]]+)\]', s)
    if shift_match:
        tags["shift"] = shift_match.group(1).strip().lower()
        text = text.replace(shift_match.group(0), "").strip()
    
    # Extract DIR if present
    dir_match = re.search(r'\[DIR=([^\]]+)\]', s)
    if dir_match:
        tags["dir"] = dir_match.group(1).strip().lower()
        text = text.replace(dir_match.group(0), "").strip()
    
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return tags, text

def _chip(label: str, cls: str = "") -> str:
    label = html.escape(label)
    cls = (" " + cls) if cls else ""
    return f'<span class="chip{cls}">{label}</span>'


def render_tag_chips(tags: dict) -> str:
    """
    Render parsed finding tags as visual chips, showing each tag category once.
    """
    if not isinstance(tags, dict) or not tags:
        return ""

    lat = tags.get("lat", "unknown").lower()
    size = tags.get("size", "unknown").lower()
    shift = tags.get("shift", "unknown").lower()
    direction = tags.get("dir", "unknown").lower()

    # Mapping with proper display
    lat_map = {
        "right": "LAT: Right",
        "left": "LAT: Left",
        "bilateral": "LAT: Bilateral",
        "unknown": "LAT: Unknown",
    }
    
    size_map = {
        "small": "SIZE: Small",
        "moderate": "SIZE: Moderate",
        "large": "SIZE: Large",
        "unknown": "SIZE: Unknown",
    }
    
    shift_map = {
        "yes": "SHIFT: Yes",
        "no": "SHIFT: No",
        "unknown": "SHIFT: Unknown",
    }
    
    dir_map = {
        "left": "DIR: Left",
        "right": "DIR: Right",
        "none": "DIR: None",
        "unknown": "DIR: Unknown",
    }

    chips = []
    
    # LAT chip (only once)
    lat_display = lat_map.get(lat, f"LAT: {lat}")
    lat_cls = "chip-strong" if lat in {"right", "left", "bilateral"} else ""
    chips.append(_chip(lat_display, lat_cls))
    
    # SIZE chip (only once)
    size_display = size_map.get(size, f"SIZE: {size}")
    size_cls = "chip-warn" if size == "large" else ("chip-strong" if size in {"small", "moderate"} else "")
    chips.append(_chip(size_display, size_cls))
    
    # SHIFT chip (only once)
    shift_display = shift_map.get(shift, f"SHIFT: {shift}")
    shift_cls = "chip-warn" if shift == "yes" else ("chip-ok" if shift == "no" else "")
    chips.append(_chip(shift_display, shift_cls))
    
    # DIR chip (only once)
    dir_display = dir_map.get(direction, f"DIR: {direction}")
    dir_cls = "chip-strong" if direction in {"left", "right"} else ""
    chips.append(_chip(dir_display, dir_cls))
    
    return f'<div class="chips">{"".join(chips)}</div>'

def render_tagged_item(s: str) -> None:
    """
    Render one finding line with optional visual tag chips.

    If structured tags are present, show them as chips before the text.
    Otherwise, render the line as a plain bullet item.
    """
    tags, text = parse_tagged_line(s)
    safe_text = html.escape(text or "").strip()

    if tags:
        chips_html = render_tag_chips(tags)
        st.markdown(
            f'<div class="tagline">'
            f'  <span class="bullet">•</span>'
            f'  {chips_html}'
            f'  <span class="finding-text">{safe_text}</span>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        # Fallback to a plain bullet line when no structured tags are detected.
        st.markdown(f"- {s}")


def remove_duplicates(items):
    """Remove duplicate string items from a list while preserving order (case-insensitive)."""
    if not isinstance(items, list):
        return items

    seen = set()
    result = []
    for item in items:
        if isinstance(item, str):
            key = item.strip().lower()
            if key not in seen:
                seen.add(key)
                result.append(item)
        else:
            result.append(item)
    return result


def show_structured_report(
    struct: dict,
    n_cases: int,
    case_paths: list[str],
    case_galleries: list[list[str]],
    case_reports=None,
    
):
    """
    Render the structured backend response into the main Streamlit analysis view.

    This includes:
    - patient findings,
    - retrieved case cards,
    - comparison summaries,
    - and the final diagnosis assessment.
    """

    patient_findings = safe_list(struct.get("patient_findings"))
    retrieved_cases = safe_list(struct.get("retrieved_cases"))
    comparison_summary = safe_list(struct.get("comparison_summary"))
    final_dx = safe_dict(struct.get("final_diagnosis_assessment"))
    case_reports = case_reports or []

    # Remove duplicates
    patient_findings = remove_duplicates(patient_findings)

    # Patient findings
    st.subheader("Patient findings")
    if patient_findings:
        for s in patient_findings:
            render_tagged_item(s)
    else:
        st.info("No patient findings returned.")

    st.markdown("---")
    st.subheader("Retrieved cases")

    for i in range(1, n_cases + 1):
        rc = next((x for x in retrieved_cases if safe_dict(x).get("case_index") == i), {})
        rc = safe_dict(rc)
        sim = rc.get("similarity_to_patient", "")
        findings = (safe_list(rc.get("findings")))

        cs = next((x for x in comparison_summary if safe_dict(x).get("case_index") == i), {})
        cs = safe_dict(cs)
        common = (safe_list(cs.get("common_with_patient")))
        diffs = (safe_list(cs.get("differences_from_patient")))
        rationale = cs.get("one_line_rationale", "")

        with st.container(border=True):
            cols = st.columns([1, 2])

            with cols[0]:
                if i - 1 < len(case_paths):
                    st.image(case_paths[i - 1], use_container_width=True)

                #st.markdown(render_badge(sim))
                st.markdown(render_badge(sim), unsafe_allow_html=True)

                gal = case_galleries[i - 1] if (i - 1) < len(case_galleries) else []
                if gal and len(gal) > 1:
                    with st.expander(f"Show all images for report ({len(gal)})", expanded=False):
                        ncols = 3
                        for start in range(0, len(gal), ncols):
                            row = st.columns(ncols)
                            for j, pth in enumerate(gal[start:start + ncols]):
                                with row[j]:
                                    st.image(pth, use_container_width=True)
                elif not gal:
                    st.caption("No extra images for this report.")

            with cols[1]:
                st.markdown(f"### Case {i}")
                # Optional: show the retrieved case report text inside the case card
                #if case_reports is None:
                #    case_reports = []

                rep_text = case_reports[i - 1] if (i - 1) < len(case_reports) else ""
                if isinstance(rep_text, str) and rep_text.strip():
                    with st.expander("Case report", expanded=False):
                        st.write(rep_text)

                #st.markdown("**Findings**")
                st.markdown("<div class='case-section-title'>Findings</div>", unsafe_allow_html=True)

                if findings:
                    for f in findings:
                        render_tagged_item(f)
                else:
                    st.write("—")

                #st.markdown("**Common with patient**")
                st.markdown("<div class='case-section-title'>Common with patient</div>", unsafe_allow_html=True)
                if common:
                    for c in common:
                        render_tagged_item(c)
                else:
                    st.write("—")

                #st.markdown("**Differences from patient**")
                st.markdown("<div class='case-section-title'>Differences from patient</div>", unsafe_allow_html=True)

                if diffs:
                    for d in diffs:
                        render_tagged_item(d)
                else:
                    st.write("—")

                if isinstance(rationale, str) and rationale.strip():
                    st.markdown("<div class='case-section-title'>Rationale</div>", unsafe_allow_html=True)
                    st.write(rationale)
    st.markdown("---")
    st.subheader("Final diagnosis assessment")

    conf = final_dx.get("confidence", "")
    text = final_dx.get("text", "")


    st.markdown(
        f"<div class='final-conf-row'>Confidence: {render_confidence(conf)}</div>",
        unsafe_allow_html=True
    )


    if isinstance(text, str) and text.strip():
        st.write(text)
    else:
        st.info("No final diagnosis text returned.")


# ====== Backend system initialization ======
@st.cache_resource
def get_rag_system():
    """Return a cached backend system instance for the Streamlit session."""

    from backend.config import AppConfig
    config = AppConfig()
    config.retrieval.weight_text2image = 0.25
    config.retrieval.weight_image2image = 0.25
    return get_system(config)


# ====== Run button ======
run_btn = st.button("Run Multimodal Analysis 🧠", type="primary", use_container_width=True)

if run_btn:

    if not query_text.strip():
        st.error("Please enter a clinical summary.")
    elif uploaded_img is None:
        st.error("Please upload a chest X-ray image.")
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            patient_path = tmp.name

        img = Image.open(uploaded_img).convert("RGB")
        img.save(patient_path)
        
        st.info("Retrieving similar cases and calling language model…")

        try:
            system = get_rag_system()
            system.config.retrieval.weight_text2image = w_t2i
            system.config.retrieval.weight_image2image = w_i2i

            with st.spinner("Running retrieval and language model inference..."):
                result = system.run_multicase_analysis(
                    query_text=query_text,
                    patient_path=patient_path,
                    k=k,
                    model=model_name,
                )

        except Exception as e:
            msg = str(e)

            # Handle Ollama/VLM memory-limit failures nicely (common in local VLMs)
            if ("memory" in msg.lower() and "available" in msg.lower()) or ("memory limits" in msg.lower()):
                st.error("VLM could not run due to insufficient system memory for the selected model.")
                st.info(
                    "Fix options:\n"
                    "- Select a smaller VLM in **Advanced model settings**\n"
                    "- Use a more quantized Ollama model (e.g., smaller/\"q\" variant)\n"
                    "- Close other applications to free RAM\n"
                    "- Run on a machine with more RAM"
                )

                # Show technical details in an expandable section to keep the main UI clean.
                with st.expander("Show technical details"):
                    import traceback
                    st.code(traceback.format_exc())

            else:
                import traceback
                st.error(f"Something went wrong: {e}")
                st.code(traceback.format_exc())

        else:
            left, right = st.columns([1.0, 2.0])


            with left:
                st.markdown("### 🩻 Patient Image")
                st.image(patient_path, use_container_width=True)

                # Retrieved case images are not shown here because they are already rendered
                # inside the Structured report view for each retrieved case.
            with right:
                tabs = st.tabs([
                    "Analysis",
                    "Retrieval debug",
                    "Raw output",
                ])


                with tabs[0]:
                    # Analysis
                    val = result.get("validation", {}) or {}
                    ok = bool(val.get("ok", False))
                    warnings = val.get("warnings", []) or []
                    errors = val.get("errors", []) or []

                    if not ok:
                        st.error("Structured output validation failed.")
                        if errors:
                            st.markdown("**Errors:**")
                            for e in errors:
                                st.markdown(f"- {e}")
                        if warnings:
                            st.markdown("**Warnings:**")
                            for w in warnings:
                                st.markdown(f"- {w}")
                        st.info("Showing raw model output instead (see Raw output tab).")
                    else:
                        if warnings:
                            st.warning("Structured output has warnings.")
                            with st.expander("Show warnings"):
                                for w in warnings:
                                    st.markdown(f"- {w}")

                        struct = result.get("llm_response_structured") or {}
                        n_cases = int(result.get("n_cases", len(result.get("case_paths", [])) or 0))
                      
                        show_structured_report(
                            struct,
                            n_cases=n_cases,
                            case_paths=result.get("case_paths", []),
                            case_galleries=result.get("case_galleries", []),
                            case_reports=result.get("case_reports", []) or [],
                        )




                with tabs[1]:
                    # Retrieval debug
                    dbg = result.get("retrieval_debug", []) or []
                    if not dbg:
                        st.write("No debug retrieval table available.")
                    else:
                        st.caption("Top retrieved candidates with fusion + negation-aware adjustment.")
                        st.dataframe(dbg, use_container_width=True)


                with tabs[2]:
                    # Raw output
                    raw = result.get("llm_response_raw", "")
                    st.code(raw if raw else "(empty)", language="json")

                    struct = result.get("llm_response_structured")
                    if isinstance(struct, dict):
                        with st.expander("Parsed JSON (pretty)"):
                            st.code(json.dumps(struct, indent=2), language="json")

        finally:
            try:
                os.remove(patient_path)
            except Exception:
                pass


# ====== Footer ======
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 14px;'>
    ThoraxDSS | Multimodal case-based decision support for chest X-ray interpretation
    </div>
    """,
    unsafe_allow_html=True
)