"""
Data loading, XML parsing, and cached artifact management.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Tuple, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

from .config import DataConfig


class DataLoader:
    """Load raw OpenI-style report/image data and manage cached processed artifacts."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        
    def build_openi_pairs(self) -> pd.DataFrame:
        """Parse XML reports and build image-text pairs for the OpenI-style dataset."""
        print("Building openi_pairs.csv from XML...")
        rows = []
        
        xml_files = list(self.config.reports_dir.glob("*.xml"))
        print(f"Found {len(xml_files)} XML reports")
        
        for xml_path in tqdm(xml_files, desc="Parsing reports"):
            try:
                tree = ET.parse(xml_path)

                # Parse the XML report so findings, impression text, and linked image IDs can be extracted.
                root = tree.getroot() 

                # Extract the main report sections used to form the text representation (findings and impression).
                findings = " ".join(
                    [el.text.strip() for el in root.findall(".//AbstractText[@Label='FINDINGS']") if el.text]
                )
                impression = " ".join(
                    [el.text.strip() for el in root.findall(".//AbstractText[@Label='IMPRESSION']") if el.text]
                )
                full_text = (findings + " " + impression).strip()
                
                
                # A single report may reference multiple associated images.
                # Extract image IDs
                image_ids = [el.attrib["id"] for el in root.findall(".//parentImage") if el.attrib.get("id")] 
                
                for img_id in image_ids: # For each id
                    img_file = self.config.images_dir / f"{img_id}.png" # the corresponding PNG file {id}.png
                    if img_file.exists():    
                        rows.append({
                            "report": xml_path.name,
                            "image": img_file.name,
                            "findings": findings,
                            "impression": impression,
                            "text": full_text,
                        })
            except Exception as e:
                print(f"[WARN] {xml_path.name}: {e}")
        
        df = pd.DataFrame(rows)
        print(f"Created {len(df)} image-text pairs")
        df.to_csv(self.config.csv_path, index=False, encoding="utf-8")
        print(f"Saved to {self.config.csv_path}")
        
        return df
    
    def load_or_build_data(self) -> pd.DataFrame:
        """
        Load the working dataframe using a cache-first strategy.

        - If df_sample.parquet exists, load and return it directly.
        - Otherwise, load the CSV if it exists; if not, rebuild it from the XML reports.
        - In the non-cached path, return the full dataframe before sampling.
        """
        # Prefer the cached sampled dataframe when available.
        df_sample = self.load_df_sample_if_exists()
        if df_sample is not None:
            return df_sample

        # Otherwise load or rebuild the full dataframe.
        if self.config.csv_path.exists():
            print(f"Loading existing {self.config.csv_path}...")
            df = pd.read_csv(self.config.csv_path)
        else:
            df = self.build_openi_pairs()

        return df
    
    def prepare_sample(self, df: pd.DataFrame, use_sample: bool, sample_size: int) -> pd.DataFrame:
        """Filter invalid rows, resolve image paths, and optionally create a sampled dataframe."""
        df = df.dropna(subset=[self.config.text_col]).copy()
        df["img_path"] = df["image"].apply(lambda x: self.config.images_dir / x)
        df = df[df["img_path"].apply(lambda p: p.exists())].reset_index(drop=True)
        
        if use_sample and len(df) > sample_size:
            df_sample = df.sample(sample_size, random_state=42).reset_index(drop=True)
        else:
            df_sample = df.copy()
        
        print(f"Sample size: {len(df_sample)}")
        return df_sample
    

    def load_df_sample_if_exists(self) -> Optional[pd.DataFrame]:
        """
        Load the cached sampled dataframe if it exists.
        When available, this is used directly as the working dataframe.
        """
        if self.config.df_sample_path.exists():
            print(f"✅ Found {self.config.df_sample_path} – loading...")
            df_sample = pd.read_parquet(self.config.df_sample_path)

            # Restore img_path values from strings back to Path objects.
            if "img_path" in df_sample.columns:
                df_sample["img_path"] = df_sample["img_path"].apply(Path)

            print(f"Loaded df_sample: {len(df_sample)} rows")
            return df_sample

        return None

    def save_df_sample(self, df_sample: pd.DataFrame) -> None:
        """Save df_sample.parquet with img_path serialized as string."""
        df_to_save = df_sample.copy()
        df_to_save["img_path"] = df_to_save["img_path"].astype(str)
        df_to_save.to_parquet(self.config.df_sample_path)
        print(f"Saved df_sample.parquet -> {self.config.df_sample_path}")


    def load_precomputed_artifacts(self) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[np.ndarray]]:
        """Load cached dataframe and embedding artifacts if all required files are available."""

        # Require the full cached artifact set before treating the cache as valid.
        if not all(p.exists() for p in [
            self.config.df_sample_path,
            self.config.text_emb_path,
            self.config.img_emb_path,
            self.config.text_index_path,
            self.config.img_index_path
        ]):
            return None, None, None
        
        print("Loading precomputed artifacts...")
        df_sample = pd.read_parquet(self.config.df_sample_path)
        df_sample["img_path"] = df_sample["img_path"].apply(Path)
        
        text_emb = np.load(self.config.text_emb_path)
        image_emb = np.load(self.config.img_emb_path)
        
        print(f"Loaded df_sample: {len(df_sample)} rows")
        print(f"Text embeddings: {text_emb.shape}")
        print(f"Image embeddings: {image_emb.shape}")
        
        return df_sample, text_emb, image_emb
    
    def save_artifacts(self, df_sample: pd.DataFrame, text_emb: np.ndarray, image_emb: np.ndarray)-> None:
        """Save the sampled dataframe and precomputed embeddings for future runs."""
        print("Saving precomputed artifacts...")
        
        df_to_save = df_sample.copy()
        df_to_save["img_path"] = df_to_save["img_path"].astype(str)
        df_to_save.to_parquet(self.config.df_sample_path)
        
        np.save(self.config.text_emb_path, text_emb)
        np.save(self.config.img_emb_path, image_emb)
        
        print("Artifacts saved successfully.")