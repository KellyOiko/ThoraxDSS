"""
Embedding and indexing utilities for multimodal retrieval.

This module provides:
- text encoding for report retrieval,
- image and text-query encoding for image retrieval,
- FAISS-based indexing for nearest-neighbor search.
"""
import torch
import numpy as np
from PIL import Image
from typing import List
from pathlib import Path

from sentence_transformers import SentenceTransformer
import open_clip
import faiss

from .config import ModelConfig
from tqdm import tqdm
import faiss

class TextEncoder:
    """Encode report text into dense embeddings using a sentence-transformer model."""
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = SentenceTransformer(config.text_model_name)

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        **kwargs
    ) -> np.ndarray:
        """Encode a batch of texts into normalized float32 embeddings."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=normalize_embeddings,
            **kwargs
        ).astype(np.float32)

class ImageEncoder:
    """Encode chest X-ray images and retrieval-oriented text queries using BiomedCLIP."""
    
    def __init__(self, config: ModelConfig):
        self.config = config

        # Load the vision-language model together with its preprocessing pipeline.
        self.model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            config.clip_model_name,
            device=config.device
        )
        self.model = self.model.eval().to(config.device)
        self.preprocess = preprocess_val
        self.tokenizer = open_clip.get_tokenizer(config.clip_model_name)
    
    def encode_images(self, image_paths: List[Path], batch_size: int = 32) -> np.ndarray:
        """Encode a list of image paths into normalized float32 embeddings."""
        
        embeddings = []
        
        for i in tqdm(range(0, len(image_paths), batch_size), desc='Encoding images'):
            batch_imgs = []
            for img_path in image_paths[i:i + batch_size]:
                img = Image.open(img_path).convert('RGB')
                batch_imgs.append(self.preprocess(img).unsqueeze(0))
            
            batch = torch.cat(batch_imgs, dim=0).to(self.config.device)
            
            # Encode and L2-normalize each batch so cosine-style similarity can be used with inner product search.
            with torch.no_grad():
                feats = self.model.encode_image(batch)
                feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            
            embeddings.append(feats.float().cpu().numpy())
        
        return np.vstack(embeddings).astype(np.float32)
    
    def encode_single_image(self, image_path: Path) -> np.ndarray:
        """Encode a single image into a normalized float32 embedding."""
        img = Image.open(image_path).convert('RGB')
        with torch.no_grad():
            inp = self.preprocess(img).unsqueeze(0).to(self.config.device)
            feats = self.model.encode_image(inp)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        
        return feats.float().cpu().numpy().astype(np.float32)
    
    def encode_text_for_image_retrieval(self, query: str) -> np.ndarray:
        """Encode a text query into the image-embedding space for text-to-image retrieval."""
        with torch.no_grad():

            # Use a radiology-style prompt so the text query better aligns with the image encoder space.
            tok = self.tokenizer([f'Chest X-ray showing {query}']).to(self.config.device)
            txt_feat = self.model.encode_text(tok)
            txt_feat = txt_feat / txt_feat.norm(p=2, dim=-1, keepdim=True)
        
        return txt_feat.float().cpu().numpy().astype(np.float32)


class FAISSIndexer:
    """Thin wrapper around a FAISS inner-product index for embedding retrieval."""
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
    
    def add(self, embeddings: np.ndarray):
        """Add embeddings to the index."""
        self.index.add(embeddings)
    
    def search(self, query: np.ndarray, k: int):
        """Search the index and return the top-k matches."""
        return self.index.search(query, k)
    
    def save(self, path: Path):
        """Save the FAISS index to disk."""
        faiss.write_index(self.index, str(path))
    
    @classmethod
    def load(cls, path: Path, dimension: int):
        """Load a FAISS index from disk and wrap it in an indexer instance."""
        indexer = cls(dimension)
        indexer.index = faiss.read_index(str(path))
        return indexer
