"""
Configuration objects for data paths, models, retrieval, and VLM settings.
"""
from pathlib import Path
from dataclasses import dataclass
import torch
from dataclasses import field

@dataclass
class DataConfig:
    """Data paths and loading configuration."""
    data_dir: Path = Path(__file__).parent.parent / "data"
    images_dir: Path = data_dir / "raw" / "NLMCXR_png"
    reports_dir: Path = data_dir / "raw" / "NLMCXR_reports" / "ecgen-radiology"
    processed_dir: Path = data_dir / "processed"
    csv_path: Path = processed_dir / "openi_pairs.csv"
    df_sample_path: Path = processed_dir / "df_sample.parquet"
    text_emb_path: Path = processed_dir / "text_emb.npy"
    img_emb_path: Path = processed_dir / "image_emb.npy"
    text_index_path: Path = processed_dir / "faiss_text.index"
    img_index_path: Path = processed_dir / "faiss_image.index"
    term_xlsx_path: str = "backend/data/radiology_vocabulary_final.xlsx"  # Radiology vocabulary file used by the negation/term-processing pipeline.

    text_col: str = "text"
    
    def ensure_directories(self):
        """Create required processed-data directories if they do not already exist."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        return self

@dataclass
class ModelConfig:
    """Model names and runtime device configuration."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    text_model_name: str = "pritamdeka/S-PubMedBert-MS-MARCO"
    clip_model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

@dataclass  
class RetrievalConfig:
    """Parameters controlling multimodal retrieval and score fusion."""

    default_k: int = 5
    default_overfetch: int = 50
    unique_by: str = "report"
    weight_text2image: float = 0.25
    weight_image2image: float = 0.25
    max_abs_adjustment: float = 0.80
    normalize_by_terms: bool = True

@dataclass
class VLMConfig:
    """Configuration for VLM requests, timeouts, and image preprocessing."""

    ollama_url: str = "http://localhost:11434/api/generate"
    default_model: str = "llava"

    # request / generation
    temperature: float = 0.0
    max_tokens: int = 700
    context_size: int = 768
    connect_timeout: int = 10
    read_timeout: int = 900

    # image preprocessing
    max_image_side: int = 768
    jpeg_quality: int = 85

@dataclass
class AppConfig:
    """Top-level application configuration grouping all subsystem settings."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)

    use_sample: bool = False
    sample_size: int = 2000
    random_state: int = 42
