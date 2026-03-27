"""
Multimodal retrieval system.
"""
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import pandas as pd
import numpy as np

from .config import RetrievalConfig
from .encoders import TextEncoder, ImageEncoder, FAISSIndexer
from .negation import TermConfig, TermProcessor, NegationAdjuster


class MultimodalRetriever:
    """Main retrieval class handling text→image, text→text, image→image retrieval."""

    def __init__(
        self,
        df_sample: pd.DataFrame,
        text_encoder: TextEncoder,
        image_encoder: ImageEncoder,
        text_indexer: FAISSIndexer,
        image_indexer: FAISSIndexer,
        config: RetrievalConfig,
 
        term_processor: Optional[TermProcessor] = None,
        negation_adjuster: Optional[NegationAdjuster] = None    ):


        self.df_sample = df_sample
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.text_indexer = text_indexer
        self.image_indexer = image_indexer
        self.config = config

        # Build helper mappings for report-level lookup and UI image grouping.
        self.report2idx = {rep: i for i, rep in enumerate(df_sample["report"].tolist())}
        self.report_to_paths = (
            df_sample.groupby("report")["img_path"]
            .apply(lambda s: [str(p) for p in s.tolist()])
            .to_dict()
        )

        # Reuse externally provided negation utilities when available;
        # otherwise create a default negation pipeline for retrieval-time adjustment.
        if negation_adjuster is not None:
            self._neg = negation_adjuster
        else:
            tp = term_processor or TermProcessor(TermConfig())
            self._neg = NegationAdjuster(tp)
            
    def search_text2text(
        self,
        query: str,
        k: int = 5,
        overfetch: int = 50,
        exclude_report: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Text-to-text retrieval.

        Retrieve the top text-matched reports for a query, keeping unique reports only.
        """   
        # IMPORTANT: TextEncoder already normalizes; do NOT pass normalize_embeddings again
        q_emb = self.text_encoder.encode([query]).astype(np.float32)
        D, I = self.text_indexer.search(q_emb, max(k, overfetch))

        seen = set()
        results = []

        for rank, idx in enumerate(I[0]):
            rep = self.df_sample.loc[idx, "report"]

            if exclude_report and rep == exclude_report:
                continue
            if rep in seen:
                continue

            seen.add(rep)
            results.append({
                "rank": len(results) + 1,
                "score": float(D[0][rank]),
                "report": rep,
                "text": str(self.df_sample.loc[idx, "text"]),
            })

            if len(results) == k:
                break

        return results

    def search_text2image(
        self,
        query: str,
        k: int = 6,
        overfetch: int = 50,
        unique_by: str = "report",
        exclude_report: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Text-to-image retrieval
        Retrieve the top text-to-image matches for a query.
        """
        q = self.image_encoder.encode_text_for_image_retrieval(query)
        D, I = self.image_indexer.search(q, max(k, overfetch))

        seen = set()
        results = []

        for rank, idx in enumerate(I[0]):
            rep = self.df_sample.loc[idx, "report"]
            path = str(self.df_sample.loc[idx, "img_path"])

            if exclude_report and rep == exclude_report:
                continue

            key = rep if unique_by == "report" else path
            if key in seen:
                continue

            seen.add(key)
            results.append({
                "rank": len(results) + 1,
                "score": float(D[0][rank]),
                "report": rep,
                "image": self.df_sample.loc[idx, "image"],
                "path": path,
                "text": str(self.df_sample.loc[idx, "text"]),
            })

            if len(results) == k:
                break

        return results

    def search_image2image(
        self,
        patient_img_path: str,
        k: int = 5,
        overfetch: int = 50,
        unique_by: str = "report",
        exclude_report: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Image-to-image retrieval
        Retrieve the top image-to-image matches for a patient image.
        """
        q = self.image_encoder.encode_single_image(Path(patient_img_path))
        D, I = self.image_indexer.search(q, max(k, overfetch))

        seen = set()
        results = []

        for pos, idx in enumerate(I[0]):
            rep = self.df_sample.loc[idx, "report"]
            path = str(self.df_sample.loc[idx, "img_path"])

            if exclude_report and rep == exclude_report:
                continue

            key = rep if unique_by == "report" else path
            if key in seen:
                continue

            seen.add(key)
            results.append({
                "rank": len(results) + 1,
                "score": float(D[0][pos]),
                "report": rep,
                "image": self.df_sample.loc[idx, "image"],
                "path": path,
                "text": str(self.df_sample.loc[idx, "text"]),
            })

            if len(results) == k:
                break

        return results


    def _get_text2text_scores(self, query: str, reports: List[str]) -> List[float]:
        """
        Compute text2text score for arbitrary reports by reconstructing the stored FAISS vectors.
        (IndexFlatIP supports reconstruct.)
        """
        q_emb = self.text_encoder.encode([query]).astype(np.float32)[0]
        scores: List[float] = []

        # FAISS reconstruct returns the stored vector for the selected index entry.
        index = self.text_indexer.index

        for rep in reports:
            idx = self.report2idx.get(rep)
            if idx is None:
                scores.append(0.0)
                continue
            try:
                v = np.array(index.reconstruct(int(idx)), dtype=np.float32)
                scores.append(float(v @ q_emb))
            except Exception:
                scores.append(0.0)

        return scores


    def run_query(
        self,
        query: str,
        patient_img_path: Optional[str] = None,
        k: int = 5,
        unique_by: str = "report",
        overfetch: int = 50,
        fusion: bool = True,
        exclude_report: Optional[str] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Run multimodal retrieval query.

        Returns:
            df_text2image, df_text2text, df_image2image, df_fused
        """
        # 1) Text→Image
        text2image_results = self.search_text2image(
            query, k=k, overfetch=overfetch, unique_by=unique_by, exclude_report=exclude_report
        )

        # 2) Text→Text
        text2text_results = self.search_text2text(
            query, k=k, overfetch=overfetch, exclude_report=exclude_report
        )

        # 3) Image→Image
        image2image_results: List[Dict[str, Any]] = []
        if patient_img_path:
            image2image_results = self.search_image2image(
                patient_img_path, k=k, overfetch=overfetch, unique_by=unique_by, exclude_report=exclude_report
            )

        df_text2image = pd.DataFrame(text2image_results)
        df_text2text = pd.DataFrame(text2text_results)
        df_image2image = pd.DataFrame(image2image_results)

        df_fused = None
        if fusion and (text2image_results or image2image_results or text2text_results):
            df_fused = self._fuse_results(df_text2image, df_text2text, df_image2image, query)

        return df_text2image, df_text2text, df_image2image, df_fused

    def _fuse_results(
        self,
        df_text2image: pd.DataFrame,
        df_text2text: pd.DataFrame,
        df_image2image: pd.DataFrame,
        query: str
    ) -> pd.DataFrame:
        """Fuse modality-specific retrieval results using report-level union, weighted scoring, and negation-aware text adjustment."""

        # Normalize empty inputs so downstream merging logic can assume
        # a consistent dataframe schema for each retrieval modality.
        df_t2i = df_text2image.copy() if not df_text2image.empty else pd.DataFrame(
            columns=["rank", "score", "report", "image", "path", "text"]
        )
        df_i2i = df_image2image.copy() if not df_image2image.empty else pd.DataFrame(
            columns=["rank", "score", "report", "image", "path", "text"]
        )
        df_t2t = df_text2text.copy() if not df_text2text.empty else pd.DataFrame(
            columns=["rank", "score", "report", "text"]
        )

        # Rename modality-specific score columns so they remain distinguishable after merging.
        if not df_t2i.empty:
            df_t2i = df_t2i.rename(columns={"score": "score_text2image"})
        if not df_i2i.empty:
            df_i2i = df_i2i.rename(columns={"score": "score_image2image"})
        if not df_t2t.empty:
            df_t2t = df_t2t.rename(columns={"score": "score_text2text_initial"})


        # Build the candidate set as the union of reports returned by any modality.
        all_reports = pd.concat(
            [
                df_t2i[["report"]] if "report" in df_t2i.columns else pd.DataFrame(columns=["report"]),
                df_i2i[["report"]] if "report" in df_i2i.columns else pd.DataFrame(columns=["report"]),
                df_t2t[["report"]] if "report" in df_t2t.columns else pd.DataFrame(columns=["report"]),
            ],
            axis=0,
            ignore_index=True,
        ).drop_duplicates("report")

        df_base = all_reports

        # Merge modality-specific scores and representative paths into a single report-level table.
        if not df_t2i.empty:
            df_base = df_base.merge(
                df_t2i[["report", "score_text2image", "image", "path"]],
                on="report",
                how="left",
            )
        else:
            df_base["score_text2image"] = 0.0
            df_base["path"] = None

        if not df_i2i.empty:
            df_base = df_base.merge(
                df_i2i[["report", "score_image2image", "image", "path"]],
                on="report",
                how="left",
                suffixes=("_t2i", "_i2i"),
            )
        else:
            df_base["score_image2image"] = 0.0

        if not df_t2t.empty:
            df_base = df_base.merge(
                df_t2t[["report", "score_text2text_initial", "text"]],
                on="report",
                how="left",
            )
        else:
            df_base["score_text2text_initial"] = 0.0

        # Ensure expected path columns exist after merging, regardless of which modalities were present.
        if "path_t2i" not in df_base.columns and "path" in df_base.columns:
            df_base = df_base.rename(columns={"path": "path_t2i"})
        if "path_i2i" not in df_base.columns:
            # If i2i merge didn't happen, create it
            df_base["path_i2i"] = None

        # Preserve all associated image paths for each report so the UI can display galleries.
        df_base["all_image_paths"] = df_base["report"].map(self.report_to_paths).apply(
            lambda x: x if isinstance(x, list) else []
        )
        df_base["n_images"] = df_base["all_image_paths"].apply(len)

        # Choose a single representative image path for plotting or preview.
        def _choose_image_path(row):
            if pd.notnull(row.get("path_i2i")) and str(row.get("path_i2i")).strip():
                return row.get("path_i2i")
            if pd.notnull(row.get("path_t2i")) and str(row.get("path_t2i")).strip():
                return row.get("path_t2i")
            paths = self.report_to_paths.get(row.get("report", ""), [])
            return str(paths[0]) if paths else None

        df_base["image_path_for_plot"] = df_base.apply(_choose_image_path, axis=1)

        # Recover the full report text for negation-aware scoring and stable downstream display.
        df_reports = (
            self.df_sample[["report", "text"]]
            .drop_duplicates("report")
            .rename(columns={"text": "report_text_full"})
        )
        df_base = df_base.merge(df_reports, on="report", how="left")

        # Recompute text-to-text scores for the full candidate union so every fused report has a comparable text score.
        df_base["score_text2text"] = self._get_text2text_scores(query, df_base["report"].astype(str).tolist())

        # Replace missing modality scores with zeros before weighted fusion.
        for col in ["score_text2image", "score_image2image", "score_text2text"]:
            if col not in df_base.columns:
                df_base[col] = 0.0
            df_base[col] = df_base[col].fillna(0.0)

        # Derive modality weights, assigning the remaining mass to text-to-text retrieval.
        w_t2i = max(0.0, float(self.config.weight_text2image))
        w_i2i = max(0.0, float(self.config.weight_image2image))
        w_t2t = max(0.0, 1.0 - w_t2i - w_i2i)

        # 1. Compute the raw fused score without negation adjustment for debugging.
        df_base["score_fused_raw"] = (
            w_t2i * df_base["score_text2image"]
            + w_i2i * df_base["score_image2image"]
            + w_t2t * df_base["score_text2text"]
        )

        # 2. Compute the negation-aware adjustment using report text only.
        def _neg_adj(txt: Any) -> float:
            return self._neg.compute_adjustment(
                patient_text=query,
                report_text=str(txt) if txt is not None else "",
                max_abs=float(self.config.max_abs_adjustment),
                normalize_by_terms=bool(self.config.normalize_by_terms),
            )

        df_base["negation_adjustment"] = df_base["report_text_full"].apply(_neg_adj)

        # 3. Apply the negation adjustment only to the text-to-text component.
        df_base["score_text2text_adj"] = df_base["score_text2text"] + df_base["negation_adjustment"]

        # 4. Compute the final fused score using the adjusted text-to-text signal.
        df_base["score_fused"] = (
            w_t2i * df_base["score_text2image"]
            + w_i2i * df_base["score_image2image"]
            + w_t2t * df_base["score_text2text_adj"]
        )

        # Rank reports by the final fused score and assign retrieval ranks.
        df_fused = df_base.sort_values("score_fused", ascending=False).reset_index(drop=True)
        df_fused["rank"] = df_fused.index + 1

        return df_fused

