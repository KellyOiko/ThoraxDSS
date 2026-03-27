"""
Main application class for the multimodal RAG pipeline.

This class orchestrates the full end-to-end workflow:
1. Load or build the dataset and retrieval indices.
2. Retrieve top-k similar prior cases from multimodal evidence.
3. Run patient-only VLM analysis to extract patient-side findings.
4. Build deterministic negation-aware comparisons against retrieved reports.
5. Run pairwise VLM comparison for each retrieved case.
6. Synthesize a final assessment and validate the output structure.

The class is intentionally designed as the central coordinator of the system,
while lower-level components handle loading, encoding, retrieval, negation logic,
VLM interaction, and JSON validation.
"""

from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import numpy as np


from .config import AppConfig
from .data_loader import DataLoader
from .encoders import TextEncoder, ImageEncoder, FAISSIndexer
from .retrieval import MultimodalRetriever
from .negation import TermProcessor, NegationAdjuster, TermConfig
from .vlm_handler import VLMHandler
from .validation import JSONParser, JSONValidator
import re
import json as jsonlib
import hashlib


class MultimodalRAGSystem:
    """Main multimodal RAG system."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self._initialize_components()
    
    def _initialize_components(self):
        """
        Initialize all major system components.

        The setup order is important:
        - load data access and encoding components,
        - load or build the dataset and vector indices,
        - initialize negation-aware term processing,
        - initialize retrieval, VLM handling, and validation.
        """
        self.data_loader = DataLoader(self.config.data)

        self.text_encoder = TextEncoder(self.config.model)
        self.image_encoder = ImageEncoder(self.config.model)

        self._load_or_build_data()

        # Initialize the term vocabulary and negation-aware comparison utilities
        # before constructing the retriever, so retrieval-time adjustment can use them.
        term_config = TermConfig(
            vocab_xlsx_path=getattr(self.config.data, "term_xlsx_path", None)
        )
        self.term_processor = TermProcessor(term_config)
        self.negation_adjuster = NegationAdjuster(self.term_processor)

        print("TERM LOAD WARNING:", term_config.load_warning)
        print("TERMS:", len(term_config.terms))
        print("SYN MAP:", len(term_config.synonyms_map))

  
  
        self.retriever = MultimodalRetriever(
            df_sample=self.df_sample,
            text_encoder=self.text_encoder,
            image_encoder=self.image_encoder,
            text_indexer=self.text_indexer,
            image_indexer=self.image_indexer,
            config=self.config.retrieval,
            
            term_processor=self.term_processor,
            negation_adjuster=self.negation_adjuster,
        )

        self.vlm_handler = VLMHandler(self.config.vlm)
        self.validator = JSONValidator()
    
    def _load_or_build_data(self):
        """
        Load precomputed dataset artifacts and FAISS indices when available.

        If cached artifacts are not found, rebuild the pipeline state from scratch:
        - load the raw dataset,
        - optionally prepare a working sample,
        - encode text and images,
        - build FAISS indices,
        - save all generated artifacts for future runs.
        """

        # First try to restore a previously prepared dataset and embeddings.
        artifacts = self.data_loader.load_precomputed_artifacts()

        if artifacts[0] is not None:
            self.df_sample, self.text_emb, self.image_emb = artifacts

            # Infer embedding dimensions from the loaded arrays so the FAISS indices
            # can be reloaded without hardcoding dimensionality.
            text_dim = int(self.text_emb.shape[1])
            img_dim = int(self.image_emb.shape[1])

            # Reload the FAISS indices using the dimensions inferred from the cached embeddings.
            self.text_indexer = FAISSIndexer.load(
                self.config.data.text_index_path,
                dimension=text_dim
            )
            self.image_indexer = FAISSIndexer.load(
                self.config.data.img_index_path,
                dimension=img_dim
            )
        else:
            # No cached artifacts were found, so rebuild the working dataset,
            # compute embeddings, and create fresh retrieval indices.
            print("Precomputed artifacts not found - building from scratch...")

            # DataLoader may return either a full dataframe or an already prepared sample.
            df_or_sample = self.data_loader.load_or_build_data()

            # If it is already sampled, reuse it directly; otherwise prepare a sample here.
            if isinstance(df_or_sample, pd.DataFrame) and "img_path" in df_or_sample.columns:
                self.df_sample = df_or_sample.reset_index(drop=True)
            else:
                df = df_or_sample
                self.df_sample = self.data_loader.prepare_sample(
                    df, self.config.use_sample, self.config.sample_size
                )
                # Save df_sample.parquet for future runs 
                self.data_loader.save_df_sample(self.df_sample)

            
            # Encode texts
            print("Encoding texts...")
            texts = self.df_sample[self.config.data.text_col].astype(str).tolist()
            self.text_emb = self.text_encoder.encode(texts)
            
            # Create text index
            self.text_indexer = FAISSIndexer(self.text_emb.shape[1])
            self.text_indexer.add(self.text_emb)
            
            # Encode images
            print("Encoding images...")
            img_paths = self.df_sample["img_path"].tolist()
            self.image_emb = self.image_encoder.encode_images(img_paths)
            
            # Create image index
            self.image_indexer = FAISSIndexer(self.image_emb.shape[1])
            self.image_indexer.add(self.image_emb)
            
            # Save artifacts
            self.data_loader.save_artifacts(self.df_sample, self.text_emb, self.image_emb)
            self.text_indexer.save(self.config.data.text_index_path)
            self.image_indexer.save(self.config.data.img_index_path)
    

    _PLACEHOLDER_RE = re.compile(r"\bX{2,}\b", flags=re.IGNORECASE)

    def _clean_placeholders(self, s: str) -> str:
        """Remove XXXX-like placeholders safely and tidy spacing/punctuation."""
        if not isinstance(s, str) or not s:
            return ""

        s = self._PLACEHOLDER_RE.sub("", s)          # Remove placeholder tokens such as XXXX or XXXXX.
        s = re.sub(r"\s{2,}", " ", s)                # Collapse repeated whitespace.
        s = re.sub(r"\s+([,.;:])", r"\1", s)         # Remove spacing artifacts before punctuation.
        return s.strip()

    def run_retrieval(
        self,
        query: str,
        patient_img_path: str,
        k: int = 3,
        exclude_report: Optional[str] = None
    ) -> Tuple[List[str], List[str], Optional[pd.DataFrame]]:
        """
        Run multimodal retrieval and return the selected case image paths,
        cleaned report texts, and the fused retrieval dataframe.
        """
  
        # Run query
        _, _, _, df_fused = self.retriever.run_query(
            query=query,
            patient_img_path=patient_img_path,
            k=max(k, 10),
            unique_by="report",
            overfetch=50,
            fusion=True,
            exclude_report=exclude_report
        )
        
        if df_fused is None or df_fused.empty:
            return [], [], df_fused
        
        # Filter out excluded report
        if exclude_report:
            df_fused = df_fused[df_fused["report"] != exclude_report].reset_index(drop=True)
            df_fused["rank"] = np.arange(1, len(df_fused) + 1)
        
        # Extract top k cases
        case_paths = []
        case_reports = []
        
        for _, row in df_fused.head(k).iterrows():
            # Resolve the representative image path for the retrieved case.
            path = row.get("image_path_for_plot") or row.get("path_i2i") or row.get("path_t2i")
            if not path:
                # Fall back to the sampled dataframe if the fused retrieval output
                # does not expose a direct image path for this report.
                matches = self.df_sample[self.df_sample["report"] == row["report"]]
                path = str(matches.iloc[0]["img_path"]) if len(matches) > 0 else None
            
            # Resolve the full report text for the retrieved case.
            report_text = row.get("report_text_full")
            if not report_text:
                matches = self.df_sample[self.df_sample["report"] == row["report"]]
                report_text = str(matches.iloc[0][self.config.data.text_col]) if len(matches) > 0 else ""

            report_text = self._clean_placeholders(str(report_text))

            case_paths.append(str(path))
            case_reports.append(report_text)

        return case_paths, case_reports, df_fused
    

    # Helper utilities for cleaning and filtering comparison outputs.
    def _negation_scrub_lists(
        self,
        common_list: List[str],
        diffs_list: List[str],
        patient_text: str,
        report_text: str,
    ) -> Tuple[List[str], List[str]]:
        """
        Filter common and difference phrase lists using negation-aware polarity maps.

        The goal is to preserve phrases that are consistent with the patient/report
        polarity interpretation while removing misleading overlap or difference
        phrases caused by negated findings or weak non-diagnostic terms.
        """
        p_map = self.term_processor.build_polarity_map(patient_text) or {}
        r_map = self.term_processor.build_polarity_map(report_text) or {}

        IGNORE = {
            "large","moderate","small","right","left","bilateral","unknown",
            "lung","lungs","thorax","chest","heart","mediastinum","spine","hilar","hilum",
            "apex","upper lobe","lower lobe","lobe","shift"
        }

        def _clean(xs: Any) -> List[str]:
            return [x.strip() for x in (xs or []) if isinstance(x, str) and x.strip()]

        def _decisive(term: str) -> bool:
            t = str(term).strip().lower()
            return bool(t) and (t not in IGNORE)

        def _contains(phrase_l: str, term: str) -> bool:
            t = str(term).strip().lower()
            if not t or t in IGNORE:
                return False
            # Approximate word-boundary matching.
            return re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", phrase_l) is not None

        ALL_TERMS = [t for t in set(list(p_map.keys()) + list(r_map.keys())) if _decisive(t)]

        def _matched_terms(phrase: str) -> List[str]:
            s = phrase.lower()
            return [t for t in ALL_TERMS if _contains(s, t)]

        def _keep_common(phrase: str) -> bool:
            mts = _matched_terms(phrase)
            if not mts:
                return True   # Keep unmatched free-text phrases instead of discarding them.
                              # This avoids over-filtering when no canonical term is detected.
            return any(p_map.get(t, 0) == +1 and r_map.get(t, 0) == +1 for t in mts)

        def _keep_diff(phrase: str) -> bool:
            mts = _matched_terms(phrase)
            if not mts:
                return True   # Keep unmatched free-text phrases instead of discarding them.
            if any(r_map.get(t, 0) == -1 for t in mts):
                return False
            return any(r_map.get(t, 0) == +1 and p_map.get(t, 0) != +1 for t in mts)

        common_out = [x for x in _clean(common_list) if _keep_common(x)]
        diffs_out  = [x for x in _clean(diffs_list)  if _keep_diff(x)]

        return common_out[:3], diffs_out[:5]



    def _detect_case_findings_conflicts(self, case_findings: List[str], report_text: str) -> List[str]:
        """
        Detect conflicts between pairwise case findings and the retrieved case report.

        A conflict is flagged when a finding phrase appears to assert a term that the
        report-side polarity map marks as negated. The method only reports conflicts;
        it does not rewrite or suppress the original findings.
        """

        IGNORE = {
            "large","moderate","small","right","left","bilateral","unknown",
            "lung","lungs","thorax","chest","heart","mediastinum","spine","hilar","hilum",
            "apex","upper lobe","lower lobe","lobe","shift"
        }
        r_map = self.term_processor.build_polarity_map(report_text) or {}
        neg_terms = {t for t, pol in r_map.items() if pol == -1 and str(t).strip().lower() not in IGNORE}

        def _contains_term(phrase_l: str, term: str) -> bool:
            t = str(term).strip().lower()
            if not t:
                return False
            return re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", phrase_l) is not None

        conflicts: List[str] = []
        for s in (case_findings or []):
            if not isinstance(s, str) or not s.strip():
                continue

            # Remove structured tag prefixes before matching against negated report terms.
            rest = re.sub(r'^\s*\[LAT=.*?\]\s*\[SIZE=.*?\]\s*\[SHIFT=.*?\]\s*\[DIR=.*?\]\s*', '', s).strip()
            rest_l = rest.lower()
            hit = [t for t in neg_terms if _contains_term(rest_l, t)]
            if hit:
                conflicts.append(f"Case finding conflicts with report negation: {', '.join(sorted(set(hit)))}")

        return conflicts



    def _drop_meta_phrases(self, xs: List[str]) -> List[str]:
        """Remove prompt/meta comparison phrases that do not describe clinical content."""

        bad = (
            "no mention", "not mentioned", "not stated",
            "text a", "text b", "image 1", "image 2",
            "report does not mention",
        )
        out: List[str] = []
        for x in (xs or []):
            if not isinstance(x, str):
                continue
            s = x.strip()
            if not s:
                continue
            sl = s.lower()
            if any(b in sl for b in bad):
                continue
            out.append(s)
        return out

    def _drop_negationish_phrases(self, xs: List[str]) -> List[str]:
        """Remove difference phrases phrased as absence or negation statements."""

        bad_prefix = (
            "no ", "without ", "negative for", "absence of", "absent ",
            "not present", "no evidence", "free of"
        )
        out: List[str] = []
        for x in (xs or []):
            if not isinstance(x, str):
                continue
            s = x.strip()
            if not s:
                continue
            sl = s.lower()
            if any(sl.startswith(p) for p in bad_prefix):
                continue
            out.append(s)
        return out

    def _drop_meta_in_findings(self, cf: List[str]) -> List[str]:
        """Remove prompt-derived or non-clinical fragments from case findings."""

        bad = ("no mention", "not mentioned", "text a", "text b", "report", "supported by")
        out = []
        for x in (cf or []):
            if not isinstance(x, str):
                continue
            s = x.strip()
            if not s:
                continue
            sl = s.lower()
            if any(b in sl for b in bad):
                continue
            out.append(s)
        return out

    
    def run_multicase_analysis(
        self,
        query_text: str,
        patient_path: str,
        k: int = 3,
        model: Optional[str] = None,
        exclude_report: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the full multimodal multi-case analysis pipeline.

        Workflow:
        1. Retrieve the most relevant prior cases for the input patient.
        2. Normalize the retrieved case set and prepare gallery information.
        3. Run a patient-only VLM call to extract structured patient findings.
        4. Compute deterministic negation-aware comparisons against each retrieved report.
        5. Run a pairwise VLM comparison between the patient and each retrieved case.
        6. Aggregate pairwise outputs into a final diagnosis-oriented assessment.
        7. Validate the final structured response and attach debugging metadata.

        If any later VLM stage fails or returns invalid JSON, the method can fall back
        to a deterministic response that preserves retrieval and comparison outputs.
        """

        # 1. Retrieve top candidate cases from the multimodal retriever.
        case_paths, case_reports, df_fused = self.run_retrieval(
            query=query_text,
            patient_img_path=patient_path,
            k=k,
            exclude_report=exclude_report
        )
        
        if not case_reports:
            return self._build_empty_response()
        
        # 2. Normalize the retrieved case set so all downstream stages use
        #    the same consistent case count.
        n_cases = int(min(k, len(case_paths), len(case_reports)))
        if n_cases <= 0:
            return self._build_empty_response()

        # 3. Prepare image galleries and stable case identifiers for reporting,
        #    warnings, and deterministic fallback outputs.
        case_galleries = self._prepare_case_galleries(df_fused, n_cases)

        # Trim all retrieved outputs to the normalized case count (n_cases).
        case_paths = case_paths[:n_cases]
        case_reports = case_reports[:n_cases]
        case_galleries = case_galleries[:n_cases]


        # Compute stable case identifiers early so warnings, debug traces,
        # and fallback outputs can reference the originating retrieved case.
        case_ids: List[str] = []
        if df_fused is not None and not df_fused.empty:
            case_ids = df_fused.head(n_cases)["report"].astype(str).tolist()

       
        # Ensure the identifier list always matches n_cases, even if retrieval
        # metadata is shorter than expected - for safety.
        if len(case_ids) < n_cases:
            case_ids = case_ids + [f"case{i}" for i in range(len(case_ids) + 1, n_cases + 1)]
        else:
            case_ids = case_ids[:n_cases]

        case_comparisons: List[Dict[str, Any]] = []
        neg_debug: List[Dict[str, Any]] = []
        neg_warnings: List[str] = []





        # A) PATIENT-ONLY VLM CALL 

        # 4. Run a patient-only VLM pass to obtain structured patient findings.
        #    These findings are later used both for final reasoning and for
        #    deterministic negation-aware comparison against retrieved reports.
        try:
            llm_raw_patient = self.vlm_handler.call_patient_only(
                query_text=query_text,
                patient_image_path=patient_path,
                model=model
            )
        except RuntimeError as e:
            # Re-raise the original patient-only VLM error here instead of switching
            # to deterministic fallback, so failures in this stage remain visible during debugging.
            raise RuntimeError(f"[patient-only VLM failed] {e}") from e

        
        # Parse, normalize, and validate the patient-only output before using it
        # in deterministic comparison and downstream final synthesis.
        patient_struct = JSONParser.try_parse_with_repair(llm_raw_patient)
        patient_struct = JSONParser.normalize_structure(patient_struct) if patient_struct else None

        if patient_struct is None or not isinstance(patient_struct, dict):
            return self._build_deterministic_response(
                case_paths, case_reports, case_galleries, case_comparisons,
                neg_debug, neg_warnings, df_fused, n_cases,
                "LLM patient-only output was not valid JSON"
            )

        # Validate that the patient-only response matches the expected JSON schema.
        v_patient = self.validator.validate_patient_only_json(patient_struct)
        if not v_patient.get("ok", False):
            return self._build_deterministic_response(
                case_paths, case_reports, case_galleries, case_comparisons,
                neg_debug, neg_warnings, df_fused, n_cases,
                "LLM patient-only output failed validation"
            )

        patient_findings = patient_struct.get("patient_findings", [])
        if not isinstance(patient_findings, list):
            patient_findings = []

        # Convert patient findings into plain text so term-level comparison operates
        # on semantic content rather than structured tag prefixes.
        def _patient_findings_as_text(pf: List[str]) -> str:
           # Remove structured tag prefixes so term extraction operates on the textual finding itself.
            lines = []
            for s in (pf or []):
                if not isinstance(s, str) or not s.strip():
                    continue
                s2 = re.sub(r'^\s*\[LAT=.*?\]\s*\[SIZE=.*?\]\s*\[SHIFT=.*?\]\s*\[DIR=.*?\]\s*', '', s).strip()
                if s2:
                    lines.append(s2)
            return "\n".join(lines)

        patient_text_for_terms = _patient_findings_as_text(patient_findings).strip()
        if not patient_text_for_terms:
            patient_text_for_terms = (query_text or "").strip()

        
        # Deterministic comparisons (AFTER patient-only call)
        # 5. Build deterministic term-level comparisons between the patient-side
        #    findings and each retrieved report. This provides a structured
        #    safety layer independent of pairwise VLM output.
        
        case_comparisons = []
        neg_debug = []
        neg_warnings = []
        
        for i, report_text in enumerate(case_reports, start=1):

            
            case_id = case_ids[i - 1]  


            comparison = self.negation_adjuster.compare_findings(patient_text_for_terms, report_text) or {}
            common_terms = comparison.get("common", []) or []
            patient_only = comparison.get("patient_only", []) or []
            report_only  = comparison.get("report_only", []) or []

            case_comparisons.append({
                "case_index": i,
                "common_terms": common_terms,
                "diff_terms": patient_only + report_only,
            })

            neg_debug.append({
                "case_index": i,
                "patient_map": self.term_processor.build_polarity_map(patient_text_for_terms),
                "report_map": self.term_processor.build_polarity_map(report_text),
            })

            IGNORE_HINTS = {
                "large","moderate","small","right","left","bilateral","unknown",
                "lung","lungs","thorax","chest","heart","mediastinum","spine","hilar","hilum",
                "apex","upper lobe","lower lobe","lobe","shift"
            }

            def _warning_term(w: str) -> str:
                # Extract term from: Negation mismatch: "pneumothorax" patient=present vs report=absent
                m = re.search(r'Negation mismatch:\s*"([^"]+)"', str(w))
                return (m.group(1).strip().lower() if m else "")

            case_warnings = self.negation_adjuster.get_negation_warnings(patient_text_for_terms, report_text) or []
            for warning in case_warnings:
                term = _warning_term(warning)
                if term and term in IGNORE_HINTS:
                    continue  # drop noisy warnings (left/right/large/etc)
                neg_warnings.append(f"Case {i} ({case_id}): [negation] {warning}")




        # B) PAIRWISE VLM CALL (for each retrieved case)

        # 6. For each retrieved case, run pairwise patient-vs-case VLM comparison,
        #    then clean and normalize the returned fields before packaging them.

        pair_raw_list: List[str] = []
        pair_post_list: List[str] = []
        retrieved_cases_out: List[Dict[str, Any]] = []
        comparison_summary_out: List[Dict[str, Any]] = []
        pair_warnings: List[str] = []

        DEBUG_PAIRWISE = True  # When enabled, print detailed per-case diagnostics for pairwise processing.

        for i in range(1, n_cases + 1):
            case_img = case_paths[i - 1]
            case_rep = case_reports[i - 1]
            case_id = case_ids[i - 1]

            # Pass only warnings associated with the current case into the pairwise VLM call.
            case_prefix = f"Case {i} ({case_id}):"
            case_neg_warnings = [w for w in neg_warnings if str(w).startswith(case_prefix)][:10]

            if DEBUG_PAIRWISE:
                q_clean = (patient_text_for_terms or "").strip()

                r_clean = (case_rep or "").strip()
                q_hash = hashlib.md5(q_clean.encode("utf-8", errors="ignore")).hexdigest()[:8]
                r_hash = hashlib.md5(r_clean.encode("utf-8", errors="ignore")).hexdigest()[:8]
                snippet = " ".join(r_clean.replace("\n", " ").split())[:650]

                print("\n" + "=" * 80)
                print(f"[PAIRWISE DEBUG] i={i}  case_id={case_id}")
                print(f"[PAIRWISE DEBUG] case_img={case_img}")
                print(f"[PAIRWISE DEBUG] query_text: len={len(q_clean)} md5={q_hash}")
                print(f"[PAIRWISE DEBUG] case_rep  : len={len(r_clean)} md5={r_hash}")
                print(f"[PAIRWISE DEBUG] case_snippet(650): {snippet[:220]}{'...' if len(snippet) > 220 else ''}")
                print(f"[PAIRWISE DEBUG] case_neg_warnings={len(case_neg_warnings)}")
                print("=" * 80 + "\n")
            

            # Build lightweight deterministic overlap/difference hints to pass into
            # the pairwise VLM call as additional structured context (common and differences).
            def _strip_status(s: str) -> str:
                # "pneumothorax [present]" -> "pneumothorax"
                return re.sub(r"\s*\[(present|absent|unknown)\]\s*$", "", str(s).strip(), flags=re.IGNORECASE).strip()

            #comp = self.negation_adjuster.compare_findings(query_text, case_rep) or {}
            comp = self.negation_adjuster.compare_findings(patient_text_for_terms, case_rep) or {}

            # Keep only positively asserted terms so the hints reflect concrete
            # present findings rather than negated findings.
            common_pos = []
            for x in (comp.get("common", []) or []):
                xs = str(x)
                if "[present]" in xs.lower():
                    common_pos.append(_strip_status(xs))

            diff_pos = []
            for x in (comp.get("report_only", []) or []):
                xs = str(x)
                if "[present]" in xs.lower():
                    diff_pos.append(_strip_status(xs))

            IGNORE_HINTS = {
                "large","moderate","small","right","left","bilateral","unknown",
                "lung","lungs","thorax","chest","heart","mediastinum","spine","hilar","hilum",
                "apex","upper lobe","lower lobe","lobe","shift"
            }

            def _dedupe_keep_order(xs: List[str]) -> List[str]:
                seen = set()
                out = []
                for x in xs:
                    x = str(x).strip()
                    if not x:
                        continue
                    xl = x.lower()
                    if xl in seen:
                        continue
                    seen.add(xl)
                    out.append(x)
                return out

            common_pos = [c for c in common_pos if c and c.lower() not in IGNORE_HINTS]
            diff_pos   = [d for d in diff_pos   if d and d.lower() not in IGNORE_HINTS]

            common_pos = _dedupe_keep_order(common_pos)[:12]
            diff_pos   = _dedupe_keep_order(diff_pos)[:12]



            # Run the pairwise VLM comparison for the current retrieved case.
            try:
                llm_raw_pair = self.vlm_handler.call_pairwise(
                    query_text=query_text,     
                    patient_image_path=patient_path,
                    case_image_path=case_img,
                    case_report=case_rep,
                    case_id=case_id,
                    warnings=case_neg_warnings,    # Only warnings associated with this case.
                    model=model,
                    
                    precomputed_common=common_pos,    
                    precomputed_diffs=diff_pos,        
                )
            except RuntimeError as e:
                return self._build_deterministic_response(
                    case_paths, case_reports, case_galleries, case_comparisons,
                    neg_debug, neg_warnings, df_fused, n_cases,
                    f"VLM pairwise failed for case {i}: {e}"
                )

            pair_raw_list.append(llm_raw_pair)

            # Parse, normalize, and validate the pairwise response before
            # incorporating it into the structured case-level summary.
            pair_struct = JSONParser.try_parse_with_repair(llm_raw_pair)
            pair_struct = JSONParser.normalize_structure(pair_struct) if pair_struct else None

            if pair_struct is None or not isinstance(pair_struct, dict):
                return self._build_deterministic_response(
                    case_paths, case_reports, case_galleries, case_comparisons,
                    neg_debug, neg_warnings, df_fused, n_cases,
                    f"LLM pairwise output was not valid JSON for case {i}"
                )

            v_pair = self.validator.validate_pairwise_json(pair_struct)
            if not v_pair.get("ok", False):
                print("[PAIRWISE VALIDATION FAIL]", i, v_pair.get("errors"), "RAW:", (llm_raw_pair or "")[:400])
                return self._build_deterministic_response(
                    case_paths, case_reports, case_galleries, case_comparisons,
                    neg_debug, neg_warnings, df_fused, n_cases,
                    f"LLM pairwise output failed validation for case {i}"
                )

            pair_warnings.extend([
                f"Case {i} ({case_id}): [pairwise_validation] {w}"
                for w in (v_pair.get("warnings", []) or [])
                if str(w).strip()
            ])

            # Extract the main pairwise fields from the validated VLM response.
            cf = pair_struct.get("case_findings", [])
           


            # Clean case findings for display and downstream checks by removing
            # tag-only fragments, meta text, and duplicate entries.
            def _clean_case_findings(xs: List[str]) -> List[str]:
                if not isinstance(xs, list):
                    return []
                bad_sub = (
                    "no mention", "not mentioned", "text a", "text b",
                    "image 1", "image 2", "report does not mention"
                )
                out = []
                for x in xs:
                    if not isinstance(x, str):
                        continue
                    s = x.strip()
                    if not s:
                        continue
                    sl = s.lower()

                    # Discard fragments that contain only structured tags rather than actual findings.
                    if re.fullmatch(r"(lat|size|shift|dir)\s*=?\s*[a-z]+", sl):
                        continue
                    if "[lat=" in sl or "lat=" in sl:
                        continue

                    if any(b in sl for b in bad_sub):
                        continue

                    out.append(s)

                # Deduplicate while preserving the original order.
                seen = set()
                ded = []
                for s in out:
                    sl = s.lower()
                    if sl in seen:
                        continue
                    seen.add(sl)
                    ded.append(s)
                return ded[:3]

            cf = _clean_case_findings(cf)


            pair_struct["case_findings"] = cf

            cf = self._drop_meta_in_findings(cf)

            # Pad the findings list to a fixed length so downstream consumers
            # always receive a consistent number of entries.
            while len(cf) < 3:
                cf.append("Additional supporting visual observation not specified.")
            cf = cf[:3]

            if not isinstance(cf, list):
                cf = []
            cf = [str(x).strip() for x in cf if str(x).strip()]


            conflicts = self._detect_case_findings_conflicts(cf, case_rep)
            for c in conflicts[:3]:
                pair_warnings.append(f"Case {i} ({case_id}): [image_vs_report_conflict] {c}")


            common = pair_struct.get("common_with_patient", [])
            common = common if isinstance(common, list) else []
            common = [str(x).strip() for x in common if isinstance(x, str) and x.strip()]

            diffs = pair_struct.get("differences_from_patient", [])
            diffs = diffs if isinstance(diffs, list) else []
            diffs = [str(x).strip() for x in diffs if isinstance(x, str) and x.strip()]

            # a. Remove prompt/meta phrasing from both overlap and difference lists.
            common = self._drop_meta_phrases(common)
            diffs  = self._drop_meta_phrases(diffs)

            # b. Remove negation-style phrases from differences.
            diffs  = self._drop_negationish_phrases(diffs)

            # c. Optionally apply stricter negation-aware filtering if needed.
            # common, diffs = self._negation_scrub_lists(common, diffs, patient_text_for_terms, case_rep)

            # d. Cap the final list sizes for stable downstream display.
            common = common[:3]
            diffs  = diffs[:5]

            if DEBUG_PAIRWISE:
                print("[PAIRWISE SCRUB] common_final=", common)
                print("[PAIRWISE SCRUB] diffs_final=", diffs)

            rationale = pair_struct.get("one_line_rationale", "")
            if not isinstance(rationale, str):
                rationale = ""
            rationale = rationale.strip()

            # Normalize the VLM-provided similarity label without recomputing it from scratch.
            sim = pair_struct.get("similarity_to_patient", "not similar")
            if not isinstance(sim, str):
                sim = "not similar"
            sim = sim.strip().lower()
            if sim not in {"similar", "partially similar", "not similar"}:
                sim = "not similar"  # safe normalization fallback


            # Conservative guardrail:
            # if the model predicts "not similar" but there is still at least one
            # meaningful shared finding and no explicit image-vs-report conflict,
            # soften the label to "partially similar".
            has_common = bool(common)  # common is already scrubbed/meta-dropped at this point
            has_conflict = any("[image_vs_report_conflict]" in str(w) for w in pair_warnings)

            if sim == "not similar" and has_common and not has_conflict:
                sim = "partially similar"

            # Write back the cleaned and normalized pairwise fields into the structured object.
            pair_struct["case_findings"] = cf
            pair_struct["common_with_patient"] = common
            pair_struct["differences_from_patient"] = diffs
            pair_struct["one_line_rationale"] = rationale
            pair_struct["similarity_to_patient"] = sim

            # Store a post-processed JSON snapshot for debugging and inspection.
            pair_post = jsonlib.dumps(pair_struct, ensure_ascii=False)
            pair_post_list.append(pair_post)


            retrieved_cases_out.append({
                "case_index": i,
                "findings": cf,
                
                "similarity_to_patient": sim,
            })

            comparison_summary_out.append({
                "case_index": i,
                
                "similarity_to_patient": sim,   
                "common_with_patient": common,
                "differences_from_patient": diffs,
                "one_line_rationale": rationale,
            })

        # C) FINAL VLM CALL (FINAL ASSESSMENT)

        # 7. Run the final VLM synthesis step using patient findings,
        #    pairwise summaries, negated terms, and accumulated warnings.
        #    A fallback assessment is prepared first so the pipeline can still
        #    return a usable result if the final response is invalid or incomplete.
        any_similar = any(
            isinstance(x.get("similarity_to_patient"), str)
            and x["similarity_to_patient"].strip().lower() == "similar"
            for x in retrieved_cases_out
        )
        conf = "medium" if any_similar else "low"

        dx_hint = patient_findings[0] if patient_findings else "key abnormality"
        final_text = (
            f"Primary impression is consistent with: {dx_hint}. "
            f"Pairwise comparison across retrieved cases was used to assess similarity. "
            f"Overall confidence is {conf}."
        )

        # Build the list of patient-side negated terms to provide explicit
        # polarity context to the final synthesis step.
        negated_terms = []
        try:
            patient_map = self.term_processor.build_polarity_map(patient_text_for_terms) or {}
            for term, pol in patient_map.items():
                if pol == -1:
                    negated_terms.append(str(term))
        except Exception:
            negated_terms = []

        # Aggregate warnings from deterministic comparison, pairwise validation,
        # and patient-only validation for use in the final synthesis call.
        final_warnings: List[str] = []
        try:
            if isinstance(neg_warnings, list):
                final_warnings.extend([str(w) for w in neg_warnings if str(w).strip()])

            if isinstance(pair_warnings, list):
                final_warnings.extend([str(w) for w in pair_warnings if str(w).strip()])

            if isinstance(v_patient, dict):
                w_pat = v_patient.get("warnings", [])
                if isinstance(w_pat, list):
                    final_warnings.extend([f"Patient: {w}" for w in w_pat if str(w).strip()])
        except Exception:
            final_warnings = []

        # Deduplicate warnings while preserving order, then cap the total count.
        seen = set()
        deduped = []
        for w in final_warnings:
            w = str(w).strip()
            if not w or w in seen:
                continue
            seen.add(w)
            deduped.append(w)
        final_warnings = deduped[:30]

        llm_raw_final = ""
        final_call_used_fallback = True

        # Attempt the final synthesis call and accept either a direct
        # {text, confidence} object or a nested final_diagnosis_assessment object.
        try:
            llm_raw_final = self.vlm_handler.call_final_assessment(
                query_text=query_text,
                patient_image_path=patient_path,
                patient_findings=patient_findings,
                comparison_summary=comparison_summary_out,
                negated_terms=negated_terms,
                warnings=final_warnings,
                model=model,
            )
            final_obj = JSONParser.try_parse_with_repair(llm_raw_final)

            # Accept nested output as well: {"final_diagnosis_assessment": {...}}
            if isinstance(final_obj, dict):
                nested = final_obj.get("final_diagnosis_assessment")
                if isinstance(nested, dict):
                    final_obj = nested

            final_obj = JSONParser.normalize_structure(final_obj) if final_obj else None

            if isinstance(final_obj, dict):
                t = final_obj.get("text")
                c = final_obj.get("confidence")

                ok_text = isinstance(t, str) and t.strip()
                ok_conf = isinstance(c, str) and c.strip().lower() in {"low", "medium", "high"}

                if ok_text:
                    final_text = t.strip()
                if ok_conf:
                    conf = c.strip().lower()

                final_call_used_fallback = not (ok_text and ok_conf)

        except Exception:
            final_call_used_fallback = True

        # Record when the final assessment had to fall back to the default summary.
        if final_call_used_fallback:
            pair_warnings.append(
                "FINAL: final_assessment did not return usable JSON {text, confidence}; used fallback final_text/conf."
            )

        # Assemble the final structured response object.
        llm_struct = {
            "patient_findings": patient_findings,
            "retrieved_cases": retrieved_cases_out,
            "comparison_summary": comparison_summary_out,
            "final_diagnosis_assessment": {
                "text": final_text,
                "confidence": conf,
            },
        }

        # Preserve raw VLM outputs from each stage for debugging and qualitative inspection.
        llm_raw = (
            "=== PATIENT_ONLY ===\n"
            + (llm_raw_patient or "")
            + "\n\n=== PAIRWISE_RAW ===\n"
            + "\n\n---\n\n".join(pair_raw_list)
            + "\n\n=== PAIRWISE_POST ===\n"
            + "\n\n---\n\n".join(pair_post_list)
            + "\n\n=== FINAL ===\n"
            + (llm_raw_final or "")
        )


        # 8. Validate the final structured response and attach all accumulated warnings.
        validation = self.validator.validate_multicase_json(llm_struct, n_cases)

        if isinstance(validation, dict):
            validation.setdefault("warnings", [])
            validation["warnings"].extend(neg_warnings)
            validation["warnings"].extend(pair_warnings)

        # Prepare a compact retrieval-debug view for downstream inspection.
        retrieval_debug = self._prepare_retrieval_debug(df_fused)

        return {
            "llm_response_raw": llm_raw,
            "llm_response_structured": llm_struct,
            "case_paths": case_paths,
            "case_reports": case_reports,
            "case_galleries": case_galleries,
            "case_comparisons": case_comparisons,
            "negation_debug": neg_debug,
            "negation_warnings": neg_warnings,
            "validation": validation,
            "n_cases": n_cases,
            "retrieval_debug": retrieval_debug,
        }

    def _prepare_case_galleries(self, df_fused: pd.DataFrame, k: int) -> List[List[str]]:
        """
        Prepare per-case image galleries for UI display.

        If a gallery is not directly available in the fused retrieval output,
        fall back to the retriever's report-to-path mapping.
        """
        galleries = []
        
        if df_fused is not None and not df_fused.empty:
            for _, row in df_fused.head(k).iterrows():
                gallery = row.get("all_image_paths")
                if not isinstance(gallery, list) or not gallery:
                    gallery = self.retriever.report_to_paths.get(row.get("report", ""), [])
                galleries.append(gallery)
        else:
            galleries = [[] for _ in range(k)]
        
        return galleries

    def _prepare_retrieval_debug(self, df_fused: pd.DataFrame) -> List[Dict]:
        """
        Prepare a compact retrieval-debug view for downstream inspection.

        Only columns available in the fused retrieval dataframe are included.
        """
        if df_fused is None or df_fused.empty:
            return []
        
        debug_cols = [
            "rank", "score_fused", "score_fused_raw", "negation_adjustment",
            "score_text2image", "score_image2image", "score_text2text", "report"
        ]
        
        # Keep only retrieval-debug columns that are present in the fused dataframe.
        available_cols = [c for c in debug_cols if c in df_fused.columns]
        
        return df_fused[available_cols].head(25).to_dict(orient="records")

    def _build_empty_response(self) -> Dict[str, Any]:
        """Return a consistent empty response object when retrieval yields no usable cases."""

        return {
            "llm_response_raw": "",
            "llm_response_structured": None,
            "case_paths": [],
            "case_reports": [],
            "case_galleries": [],
            "case_comparisons": [],
            "negation_debug": [],
            "negation_warnings": [],
            "validation": {
                "ok": False,
                "errors": ["No retrieved cases"],
                "warnings": [],
                "logic_ok": True
            },
            "n_cases": 0,
            "retrieval_debug": [],
        }

    def _build_deterministic_response(
        self,
        case_paths: List[str],
        case_reports: List[str],
        case_galleries: List[List[str]],
        case_comparisons: List[Dict],
        neg_debug: List[Dict],
        neg_warnings: List[str],
        df_fused: pd.DataFrame,
        k: int,
        warning_message: str
    ) -> Dict[str, Any]:
        """
        Build a deterministic fallback response when a VLM stage fails.

        This preserves retrieval outputs and deterministic comparisons so the
        pipeline can still return a structured, inspectable result.
        """
        # Build the fallback structured response using deterministic comparison outputs.
        deterministic_struct = {
            "patient_findings": [],
            "retrieved_cases": [
                {
                    "case_index": i,
                    "findings": [],
                    "similarity_to_patient": "not similar"
                }
                for i in range(1, k + 1)
            ],
            "comparison_summary": [
                {
                    "case_index": item.get("case_index"),
                    "common_with_patient": item.get("common_terms", []),
                    "differences_from_patient": item.get("diff_terms", []),
                    "one_line_rationale": "Deterministic comparison (LLM unavailable).",
                }
                for item in case_comparisons
            ],

            "final_diagnosis_assessment": {
                "text": f"LLM unavailable. {warning_message}",
                "confidence": "low",
            },
        }
        
        # Build fallback validation metadata.
        validation = {
            "ok": True,
            "errors": [],
            "warnings": [warning_message] + neg_warnings,
            "logic_ok": True
        }
        
        # Prepare retrieval-debug information for downstream inspection.
        retrieval_debug = self._prepare_retrieval_debug(df_fused)
        
        return {
            "llm_response_raw": "",
            "llm_response_structured": deterministic_struct,
            "case_paths": case_paths,
            "case_reports": case_reports,
            "case_galleries": case_galleries,
            "case_comparisons": case_comparisons,
            "negation_debug": neg_debug,
            "negation_warnings": neg_warnings,
            "validation": validation,
            "n_cases": len(case_reports),
            "retrieval_debug": retrieval_debug,
        }


# Shared singleton instance to avoid rebuilding the system on repeated access.
_system_instance = None

def get_system(config: Optional[AppConfig] = None) -> MultimodalRAGSystem:
    """Return the existing system instance or create it on first use."""
    global _system_instance
    if _system_instance is None:
        # Initialize the system only the first time.
        _system_instance = MultimodalRAGSystem(config)
    return _system_instance