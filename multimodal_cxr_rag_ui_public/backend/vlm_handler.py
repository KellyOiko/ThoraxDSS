"""
VLM prompt building, image preparation, and Ollama request handling.
"""
import base64
import json
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple

import requests
from PIL import Image

from .config import VLMConfig
import re

class ImageEncoder:
    """Handles image encoding for VLM."""

    def __init__(self, config: VLMConfig):
        self.config = config

    def encode_to_base64(self, image_path: str) -> str:
        """Resize an image if needed, convert it to JPEG, and encode it as base64."""
        im = Image.open(image_path).convert("RGB")

        w, h = im.size
        scale = max(w, h) / float(self.config.max_image_side)
        if scale > 1.0:
            im = im.resize((int(w / scale), int(h / scale)), resample=Image.BILINEAR)

        buf = BytesIO()
        im.save(buf, format="JPEG", quality=self.config.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

class VLMPromptBuilder:
    """Build prompts for the patient-only, pairwise, and final VLM stages."""

    @staticmethod
    def build_patient_only_prompt(query_text: str) -> str:
        """
        Build the patient-only prompt.

        This stage returns only the patient_findings field in a compact JSON object.
        The schema is intentionally kept stable so downstream parsing and validation
        remain compatible. Laterality, size, and shift information are encoded
        inside the first finding using a strict tag format.
        """

        prompt = f"""
    You will analyze a PATIENT chest X-ray.

    IMAGE MAPPING:
    - PATIENT: Image 1

    CLINICAL SUMMARY (PATIENT REPORT TEXT):
    {query_text}

    TASK:
    Analyze the patient image and the clinical summary.
    Return ONLY a single valid JSON object with this schema:

    {{
    "patient_findings": [
        "One short sentence per finding."
    ]
    }}

    CRITICAL OUTPUT FORMAT (must follow exactly):
    - You MUST return 4–5 DISTINCT patient findings (4–5 list items).
    - ONLY the FIRST finding string MUST start with tags in this exact order:

    "[LAT=<laterality>] [SIZE=<size>] [SHIFT=<yes/no/unknown>] [DIR=<left/right/none/unknown>] <finding sentence>"

    - Findings 2–5 MUST NOT include any tags. They must be plain short sentences.

    Where the enums are:
    - laterality (LAT): right | left | bilateral | unknown
    - size (SIZE): small | moderate | large | unknown
    - shift present (SHIFT): yes | no | unknown
    - shift direction (DIR): left | right | none | unknown

    RULES:
    - If unsure about any enum, use "unknown" (do NOT guess).
    - Tags should refer to the MAIN abnormality (e.g., pneumothorax/effusion/mass).
    - SHIFT/DIR should refer to mediastinal/tracheal shift suggesting tension physiology (if applicable).
    - Avoid unsupported findings. Be faithful to image and consistent with the clinical summary.
    - Output MUST be valid JSON, double quotes only.
    - No comments, no extra text.
    - Do NOT use placeholders like "XXXX". If uncertain, omit that finding.
    - One atomic finding per sentence.
    - Do NOT repeat the same finding in different words.

    EXAMPLE (format only):
    {{
    "patient_findings": [
        "[LAT=right] [SIZE=large] [SHIFT=yes] [DIR=left] Large right pneumothorax with mediastinal shift.",
        "Right lung shows marked hyperlucency and absent peripheral vascular markings.",
        "Collapsed right lung is visualized medially.",
        "Cardiomediastinal contours are displaced (tension physiology if applicable).",
        "No left-sided pneumothorax is identified."
    ]
    }}

    Return ONLY the JSON object.
    """
        return prompt.strip()
            




    @staticmethod
    def build_pairwise_prompt(
        query_text: str,
        case_report: str,
        case_id: str,
        warnings: Optional[List[str]] = None,
        precomputed_common: Optional[List[str]] = None,  # optional hints only
        precomputed_diffs: Optional[List[str]] = None,   # optional hints only
    ) -> str:
        """
        Build the pairwise comparison prompt for the patient image and one retrieved case.

        The prompt includes compact case-report context, optional warning signals,
        and optional overlap/difference hints to stabilize structured JSON output.
        """
                
        # Keep the case-report snippet short and stable to reduce prompt variability.
        snippet = " ".join((case_report or "").replace("\n", " ").split())[:520]

        safe_warnings = [str(w).strip() for w in (warnings or []) if str(w).strip()][:6]
        hint_common = [str(x).strip() for x in (precomputed_common or []) if str(x).strip()][:10]
        hint_diffs = [str(x).strip() for x in (precomputed_diffs or []) if str(x).strip()][:10]

        prompt = f"""
        Analyze TWO chest X-rays: PATIENT (Image 1) and RETRIEVED CASE (Image 2).

        TEXT A (patient report):
        {query_text}

        TEXT B (case report snippet, id={case_id}):
        {snippet}

        WARNINGS (only for overlap filtering; do NOT quote):
        {json.dumps(safe_warnings, ensure_ascii=False)}

        TERM HINTS (anchors only; do NOT copy as standalone items):
        overlap_terms_hint={json.dumps(hint_common, ensure_ascii=False)}
        case_only_terms_hint={json.dumps(hint_diffs, ensure_ascii=False)}

        OUTPUT RULES:
        - Output ONE JSON object ONLY. No extra text.
        - Keys must be exactly:
        case_findings, similarity_to_patient, common_with_patient, differences_from_patient, one_line_rationale.
        - All list items must be NON-EMPTY strings.
        - Do NOT output any of these anywhere: "No meaningful overlap found", "<", ">", "...", "TODO".


        case_findings (Image 2 ONLY):
        - MUST be a list of EXACTLY 3 strings.
        - Each string MUST be a short VISUAL radiology finding about Image 2.
        - ABSOLUTELY FORBIDDEN inside case_findings:
          * any tag text like "LAT=right" or "[LAT=right]"
          * meta phrases like "No mention", "Text A", "Text B", "Image 1", "Image 2", "report does not mention"

        common_with_patient / differences_from_patient (TEXT A vs TEXT B ONLY):
        - Use only clinician-friendly radiology phrases supported by the texts.
        - If no explicit support: output [] (do NOT invent).
        - Do NOT output meta phrases like "No mention of..." in these lists.

        similarity_to_patient:
        - Choose exactly one: "similar", "partially similar", "not similar" (lowercase).

        one_line_rationale:
        - Exactly 3 short sentences.
        - Must mention BOTH report-text alignment AND visual alignment.

        Return JSON:
        {{
          "case_findings": ["string","string","string"],
          "similarity_to_patient": "similar",
          "common_with_patient": ["string"],
          "differences_from_patient": ["string"],
          "one_line_rationale": "sentence. sentence. sentence."
        }}
        """.strip()



        return prompt


    # Regex helpers for compact warning parsing in prompt-building / diagnostics.

    _CASE_WARN_RE = re.compile(
        r"^Case\s+(?P<case_index>\d+)\s*\((?P<case_id>[^)]+)\):\s*\[(?P<kind>[^\]]+)\]\s*(?P<msg>.+)$"
    )

    # Optional regex helper for extracting structured negation-mismatch details.
    _NEG_MISMATCH_RE = re.compile(
        r'Negation mismatch:\s*"?(?P<term>[^"]+)"?\s+patient=(?P<p>present|absent)\s+vs\s+report=(?P<r>present|absent)',
        re.IGNORECASE
    )


    @staticmethod
    def build_final_assessment_prompt(
        query_text: str,
        patient_findings: List[str],
        comparison_summary: List[Dict[str, Any]],  # Retained for compatibility with the current pipeline.
        negated_terms: List[str],                  # Retained for compatibility with the current pipeline.
        warnings: List[str],
    ) -> str:
        """
        Final assessment step: returns ONLY the final clinician-facing assessment JSON.

        IMPORTANT:
        - This stage must return only the final {"text", "confidence"} object.
        - It uses aggregated consistency signals rather than case text, to keep the
          final synthesis compact, stable, and focused on patient-facing evidence.
        - Negated terms are treated as a quality/context signal, not as a direct penalty.
        """

        # Keep final-stage inputs compact and stable; do not expose case text at this stage.
        safe_findings = [str(f).strip() for f in (patient_findings or []) if str(f).strip()][:20]

        safe_summary = (comparison_summary or [])[:20]
        n_cases = 0
        n_similar = 0
        n_partial = 0
        n_not = 0
        try:
            n_cases = len(safe_summary)
            for s in safe_summary:
                if not isinstance(s, dict):
                    continue
                sim = s.get("similarity_to_patient")
                if isinstance(sim, str):
                    sim_l = sim.strip().lower()
                    if sim_l == "similar":
                        n_similar += 1
                    elif sim_l == "partially similar":
                        n_partial += 1
                    elif sim_l == "not similar":
                        n_not += 1
        except Exception:
            n_cases = 0
            n_similar = n_partial = n_not = 0

        # Keep negated_terms_count as a quality/context signal rather than a direct penalty driver.
        n_neg_terms = len([t for t in (negated_terms or []) if str(t).strip()])
        n_warn = len([w for w in (warnings or []) if str(w).strip()])

        return f"""
    You are producing a FINAL clinician-facing assessment for a chest X-ray system.

    YOU SEE ONLY ONE IMAGE HERE:
    - Image 1 = PATIENT chest X-ray

    OUTPUT JSON ONLY (one JSON object). No markdown. No extra text.
    If you cannot comply, output exactly:
    {{"text":"", "confidence":"low"}}

    ALLOWED EVIDENCE SOURCES:
    1) CLINICAL SUMMARY text (below)
    2) PATIENT_FINDINGS list (below)
    3) Image 1 (for verification / calibration)

    CRITICAL CONTENT RULES:
    A) PRIMARY ABNORMALITIES:
    - You may state as "present/likely" ONLY abnormalities that are already mentioned in PATIENT_FINDINGS or CLINICAL SUMMARY.
    - Do NOT introduce new abnormality categories as definite.

    B) IMAGE-ONLY SUGGESTIONS (LIMITED, OPTIONAL):
    - You MAY add at most 1 sentence with up to 1–2 uncertain "image-suggested" items,
    but ONLY from this whitelist and ONLY phrased as "may suggest / possible / cannot exclude".
    - You MUST NOT state any whitelist item as present/likely unless it already appears in PATIENT_FINDINGS or CLINICAL SUMMARY.
    - If unsure, omit image-suggested items entirely.
    - If an item is NOT explicitly in PATIENT_FINDINGS/CLINICAL SUMMARY, you MUST phrase it only as "possible/may suggest" and NEVER as present/likely.

    C) MAIN FINDING CALIBRATION:
    - Set confidence="low" ONLY if:
    (a) you explicitly wrote "image-report mismatch" due to CLEAR contradiction, OR
    (b) warnings_count is very high AND the retrieved cases are broadly unsupportive (mostly not similar).
    - Do NOT set low due to "uncertainty" alone.


    INPUTS
    CLINICAL SUMMARY (patient report text):
    {query_text}

    PATIENT_FINDINGS (primary evidence):
    {json.dumps(safe_findings, ensure_ascii=False)}

    CONSISTENCY SIGNALS (aggregated only; NO case text is provided):
    cases_seen={n_cases}; similar={n_similar}; partially_similar={n_partial}; not_similar={n_not};
    negated_terms_count={n_neg_terms}; warnings_count={n_warn}

    REQUIRED "text" FORMAT (MUST follow exactly, headings exact, order exact):
    EVIDENCE:
    <1 paragraph: summarize the main abnormality + key supporting signs, grounded in PATIENT_FINDINGS + Image 1. You MAY paraphrase.>

    CASE CONSISTENCY:
    <1 paragraph: use only the aggregated counts to describe overall support. Note that partial similarity across different patients can still be supportive. No case details.>

    NEGATION CHECK:
    <1–2 sentences:
    - If negated_terms_count==0: write "No explicit negations highlighted in the patient text."
    - Else: write "Patient text contains explicit negations; this is normal in radiology reports and does NOT reduce confidence by itself. Use it only to avoid contradicting negated statements."
    Do NOT list terms. Do NOT mention any case IDs.>

    IMPRESSION:
    <1 paragraph: final impression using ONLY abnormalities from PATIENT_FINDINGS/CLINICAL SUMMARY as "likely/present".>
    <OPTIONAL: add 1 sentence with up to 1–2 "image-suggested" items from the whitelist, clearly uncertain.>
    <If mismatch exists, include the phrase "image-report mismatch".>

    CONFIDENCE RATIONALE:
    <1 paragraph: explain confidence using only: mismatch yes/no, warnings_count, and similarity counts.
    You MAY mention that explicit negations require caution, but you MUST NOT use negated_terms_count as a direct penalty.>

    CONFIDENCE RULES (MUST APPLY; DSS-STYLE, NOT OVERLY STRICT):
    - Default confidence is "medium".
    - Set confidence="low" ONLY if you explicitly wrote the exact phrase "image-report mismatch".
    - Otherwise, do NOT output "low".
    - Set confidence="high" only if Image 1 clearly supports the main abnormality AND warnings_count is minimal AND there is at least some retrieval support.

    OUTPUT JSON ONLY:
    {{
    "text": "EVIDENCE:\\n...\\n\\nCASE CONSISTENCY:\\n...\\n\\nNEGATION CHECK:\\n...\\n\\nIMPRESSION:\\n...\\n\\nCONFIDENCE RATIONALE:\\n...",
    "confidence": "low|medium|high"
    }}
    """.strip()





class VLMHandler:
    """Handles communication with Ollama VLM."""

    def __init__(self, config: VLMConfig):
        self.config = config
        self.image_encoder = ImageEncoder(config)
        

    def generate_stream(self, payload: Dict[str, Any]) -> Tuple[str, Optional[Dict]]:
        """
        Send a streaming request to Ollama and collect the final text response.

        Returns the concatenated response text together with the last streamed
        JSON object received from the server.
        """
        
        payload = dict(payload)
        payload["stream"] = True

        full_response: List[str] = []
        last_obj = None

        try:
            r = requests.post(
                self.config.ollama_url,
                json=payload,
                stream=True,
                timeout=(self.config.connect_timeout, self.config.read_timeout),
            )

            if not r.ok:
                raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text}")
 
            # Ollama streams newline-delimited JSON objects; accumulate response chunks incrementally.
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                last_obj = obj

                if obj.get("error"):
                    raise RuntimeError(
                        f"Ollama stream error: {obj.get('error')} | model={payload.get('model')}"
                    )

                chunk = obj.get("response", "")
                if chunk:
                    full_response.append(chunk)

                if obj.get("done", False):
                    break

            # Reconstruct the final text from all streamed response fragments.
            text = "".join(full_response).strip()
            if not text:
                raise RuntimeError(
                    f"Ollama returned EMPTY response | model={payload.get('model')} "
                    f"(likely crash/OOM/context/endpoint issue)"
                )

            return text, last_obj

        except requests.exceptions.Timeout:
            raise RuntimeError("Ollama request timed out")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Cannot connect to Ollama server")

    def _base_payload(self, prompt: str, images_b64: List[str], model: Optional[str]) -> Dict[str, Any]:
        """Build the base Ollama request payload shared across all VLM stages."""

        return {
            "model": model or self.config.default_model,
            "prompt": prompt,
            "images": images_b64,
            "format": "json",
            "keep_alive": 0,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "num_ctx": min(int(self.config.context_size), 2048),
            },
        }



    # ---------- Stage-specific VLM calls ----------
    def call_patient_only(
        self,
        query_text: str,
        patient_image_path: str,
        model: Optional[str] = None,
    ) -> str:
        """
        Run the patient-only VLM stage and return the raw JSON text response.

        The response is intentionally returned as raw text so it can be parsed
        later by the existing JSON parsing and validation pipeline.
        """

        images_b64 = [self.image_encoder.encode_to_base64(patient_image_path)]
        prompt = VLMPromptBuilder.build_patient_only_prompt(query_text)

        payload = self._base_payload(prompt=prompt, images_b64=images_b64, model=model)

        try:
            response, _ = self.generate_stream(payload)
            return response
        except RuntimeError as e:
            msg = str(e)

            # Preserve the explicit memory-limit handling for known Ollama failure modes.
            if "requires more system memory" in msg or "Ollama error 500" in msg:
                raise RuntimeError("VLM unavailable due to memory limits")
            
            # Propagate the exact stage-specific error for easier debugging upstream.
            raise RuntimeError(f"[patient_only] {msg}")


    def call_pairwise(
        self,
        query_text: str,
        patient_image_path: str,
        case_image_path: str,
        case_report: str,
        case_id: str,
        warnings: Optional[List[str]] = None,
        model: Optional[str] = None,
        precomputed_common: Optional[List[str]] = None,
        precomputed_diffs: Optional[List[str]] = None,
    ) -> str:
        """
        Run the pairwise VLM stage for the patient image and one retrieved case.

        Returns the raw JSON text response for downstream parsing and validation.
        """
                
        images_b64 = [
            self.image_encoder.encode_to_base64(patient_image_path),
            self.image_encoder.encode_to_base64(case_image_path),
        ]

        prompt = VLMPromptBuilder.build_pairwise_prompt(
            query_text=query_text,
            case_report=case_report,
            case_id=case_id,
            warnings=warnings,
            precomputed_common=precomputed_common,
            precomputed_diffs=precomputed_diffs,
        )

        # Lightweight prompt diagnostics for pairwise debugging.
        print(f"[PAIRWISE PROMPT CHECK][{case_id}] common={precomputed_common}", flush=True)
        print(f"[PAIRWISE PROMPT CHECK][{case_id}] diffs={precomputed_diffs}", flush=True)
        print(f"[PAIRWISE PROMPT CHECK][{case_id}] warnings={warnings}", flush=True)

        payload = self._base_payload(prompt=prompt, images_b64=images_b64, model=model)
        response, _ = self.generate_stream(payload)
        return response




    def call_final_assessment(
        self,
        query_text: str,
        patient_image_path: str,
        patient_findings: List[str],
        comparison_summary: List[Dict[str, Any]],
        negated_terms: List[str],
        warnings: List[str],  
        model: Optional[str] = None,
    ) -> str:
        """
        Run the final VLM synthesis stage and return the raw JSON text response.
        """
                
        images_b64 = [self.image_encoder.encode_to_base64(patient_image_path)]
        prompt = VLMPromptBuilder.build_final_assessment_prompt(
            query_text=query_text,
            patient_findings=patient_findings,
            comparison_summary=comparison_summary,
            negated_terms=negated_terms,
            warnings=warnings,  
        )
        payload = self._base_payload(prompt=prompt, images_b64=images_b64, model=model)
        response, _ = self.generate_stream(payload)
        return response