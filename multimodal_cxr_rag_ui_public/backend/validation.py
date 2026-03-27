"""
JSON validation and parsing utilities.
"""
import json
import re
from typing import Dict, Any, List, Optional, Tuple


class JSONParser:
    """Robust JSON parsing with repair capabilities."""

    @staticmethod
    def clean_json_text(text: str) -> str:
        """Clean JSON text by removing markdown code blocks."""
        if not text:
            return text

        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def extract_first_json_object(text: str) -> Optional[Dict]:
        """
        Extract the first valid JSON object from a text response.

        This handles nested braces correctly when they appear inside quoted strings.
        """

        if not text:
            return None

        # Remove markdown code blocks
        text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

        start = text.find("{")
        if start == -1:
            return None

        in_str = False
        esc = False
        depth = 0
        end = None

        for i in range(start, len(text)):
            ch = text[i]

            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break

        if end is None:
            return None

        candidate = text[start:end + 1].strip()
        try:
            return json.loads(candidate)
        except Exception:
            return None

    @staticmethod
    def try_parse_with_repair(text: str) -> Optional[Dict]:
        """Try multiple parsing strategies with repair."""
        if not text:
            return None

        text_clean = JSONParser.clean_json_text(text)

        # 1. Try direct JSON parsing.
        try:
            return json.loads(text_clean)
        except Exception:
            pass

        # 2. Try extracting the first balanced JSON object.
        obj = JSONParser.extract_first_json_object(text_clean)
        if obj is not None:
            return obj

        # 3. Try a simple repair pass for trailing commas.
        text_fixed = re.sub(r",\s*([}\]])", r"\1", text_clean)
        return JSONParser.extract_first_json_object(text_fixed)

    @staticmethod
    def normalize_structure(data: Dict) -> Dict:
        """Normalize parsed JSON fields such as similarity and confidence labels."""
        if not isinstance(data, dict):
            return data

        # Normalize similarity labels (multicase)
        rc = data.get("retrieved_cases", [])
        if isinstance(rc, list):
            for case in rc:
                if not isinstance(case, dict):
                    continue
                sim = case.get("similarity_to_patient")
                if isinstance(sim, str):
                    case["similarity_to_patient"] = sim.strip().lower()

        # Normalize similarity label (pairwise)
        sim2 = data.get("similarity_to_patient")
        if isinstance(sim2, str):
            data["similarity_to_patient"] = sim2.strip().lower()

        # Normalize confidence (multicase only; harmless if missing)
        fd = data.get("final_diagnosis_assessment", {})
        if isinstance(fd, dict):
            conf = fd.get("confidence")
            if isinstance(conf, str):
                fd["confidence"] = conf.strip().lower()

        return data


class JSONValidator:
    """Validates structured JSON output."""

    # Allowed values
    ALLOWED_SIMILARITY = {"similar", "partially similar", "not similar"}
    ALLOWED_CONFIDENCE = {"low", "medium", "high"}


    @staticmethod
    def validate_patient_only_json(data: Dict) -> Dict[str, Any]:
        """Validate the patient-only JSON structure."""

        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(data, dict):
            return {"ok": False, "errors": ["Not a dictionary"], "warnings": [], "logic_ok": False}

        if "patient_findings" not in data:
            errors.append("Missing required key: patient_findings")

        pf = data.get("patient_findings", [])

        # Validate the patient_findings list using permissive tag checking.
        JSONValidator._check_tagged_finding_list(
            items=pf,
            warnings=warnings,
            errors=errors,
            where="patient_findings",
            require_tags=False,
        )

        logic_ok = len(errors) == 0
        return {"ok": logic_ok, "errors": errors, "warnings": warnings, "logic_ok": logic_ok}

    
    @staticmethod
    def _check_tagged_finding_list(
        items: Any,
        warnings: List[str],
        errors: List[str],
        where: str,
        require_tags: bool = False,
    ) -> None:
        """
        Validate a list of finding strings that may begin with structured tags such as:
        [LAT=right] [SIZE=large] [SHIFT=yes] [DIR=left] Finding text...

        Tag parsing is intentionally permissive by default:
        - missing or invalid tags do not trigger warnings
        - placeholder-style text is ignored at this stage
        - hard errors are reserved for wrong container/item types, or when require_tags=True
        """

        if not isinstance(items, list):
            errors.append(f"{where} must be a list")
            return

        
        tag_re = re.compile(
            r'^\s*\[LAT=(?P<lat>right|left|bilateral|unknown)\]\s*'
            r'\[SIZE=(?P<size>small|moderate|large|unknown)\]\s*'
            r'\[SHIFT=(?P<shift>yes|no|unknown)\]\s*'
            r'\[DIR=(?P<dir>left|right|none|unknown)\]\s*'
            r'(?P<text>.+)$',
            re.IGNORECASE
        )

        for j, s in enumerate(items, start=1):
            if not isinstance(s, str):
                errors.append(f"{where}[{j}] must be a string")
                continue

            if require_tags and not tag_re.match(s.strip()):
                errors.append(f"Missing/invalid tags in {where}[{j}]")


    @staticmethod
    def validate_pairwise_json(data: Dict) -> Dict[str, Any]:
        """Validate the pairwise comparison JSON structure."""

        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(data, dict):
            return {"ok": False, "errors": ["Not a dictionary"], "warnings": [], "logic_ok": False}

        required_keys = [
            "case_findings",
            "similarity_to_patient",
            "common_with_patient",
            "differences_from_patient",
            "one_line_rationale",
        ]
        for key in required_keys:
            if key not in data:
                errors.append(f"Missing required key: {key}")

        # Basic types
        if "one_line_rationale" in data and not isinstance(data.get("one_line_rationale"), str):
            errors.append("one_line_rationale must be a string")

        # Tagged lists
        JSONValidator._check_tagged_finding_list(
            items=data.get("case_findings", []),
            warnings=warnings,
            errors=errors,
            where="case_findings",
            require_tags=False,
        )
        JSONValidator._check_tagged_finding_list(
            items=data.get("common_with_patient", []),
            warnings=warnings,
            errors=errors,
            where="common_with_patient",
            require_tags=False,
        )
        JSONValidator._check_tagged_finding_list(
            items=data.get("differences_from_patient", []),
            warnings=warnings,
            errors=errors,
            where="differences_from_patient",
            require_tags=False,
        )


        # Enum check (strict)
        sim = data.get("similarity_to_patient")
        if not isinstance(sim, str):
            errors.append("similarity_to_patient must be a string")
        else:
            sim_norm = sim.strip().lower()
            if sim_norm not in JSONValidator.ALLOWED_SIMILARITY:
                errors.append(f"Unexpected similarity label: {sim}")

 
        logic_ok = len(errors) == 0
        return {"ok": logic_ok, "errors": errors, "warnings": warnings, "logic_ok": logic_ok}

    @staticmethod
    def validate_multicase_json(data: Dict, expected_cases: int) -> Dict[str, Any]:
        """Validate the full multicase JSON structure."""

        errors: List[str] = []
        warnings: List[str] = []

        if not isinstance(data, dict):
            return {"ok": False, "errors": ["Not a dictionary"], "warnings": [], "logic_ok": False}

        required_keys = [
            "patient_findings",
            "retrieved_cases",
            "comparison_summary",
            "final_diagnosis_assessment"
        ]
        for key in required_keys:
            if key not in data:
                errors.append(f"Missing required key: {key}")

        rc = data.get("retrieved_cases", [])
        cs = data.get("comparison_summary", [])

        if not isinstance(rc, list):
            errors.append("retrieved_cases must be a list")
            rc = []
        if not isinstance(cs, list):
            errors.append("comparison_summary must be a list")
            cs = []

        if len(rc) != expected_cases:
            errors.append(f"retrieved_cases length {len(rc)} != expected {expected_cases}")
        if len(cs) != expected_cases:
            errors.append(f"comparison_summary length {len(cs)} != expected {expected_cases}")

        # Validate patient_findings with permissive tag checking.
        JSONValidator._check_tagged_finding_list(
            items=data.get("patient_findings", []),
            warnings=warnings,
            errors=errors,
            where="patient_findings",
            require_tags=False,
        )

        # Validate retrieved case indices, similarity labels, and findings.
        for i, case in enumerate(rc, start=1):
            if not isinstance(case, dict):
                errors.append(f"retrieved_cases[{i}] must be a dict")
                continue

            if case.get("case_index") != i:
                errors.append(f"retrieved_cases case_index mismatch at position {i}")

            sim = case.get("similarity_to_patient")
            if isinstance(sim, str) and sim.strip().lower() not in JSONValidator.ALLOWED_SIMILARITY:
                warnings.append(f"Unexpected similarity label: {sim}")

            JSONValidator._check_tagged_finding_list(
                items=case.get("findings", []),
                warnings=warnings,
                errors=errors,
                where=f"retrieved_cases[{i}].findings",
                require_tags=False,
            )

        # Validate comparison summaries, including indices, rationale text, and list fields.
        for i, item in enumerate(cs, start=1):
            if not isinstance(item, dict):
                errors.append(f"comparison_summary[{i}] must be a dict")
                continue

            if item.get("case_index") != i:
                errors.append(f"comparison_summary case_index mismatch at position {i}")

            rationale = item.get("one_line_rationale", "")
            if isinstance(rationale, str):
                for other in range(1, expected_cases + 1):
                    if other != i and (f"Case {other}" in rationale or f"case {other}" in rationale):
                        warnings.append(f"Possible cross-case leakage in rationale of case {i}")
            else:
                errors.append(f"comparison_summary[{i}].one_line_rationale must be a string")

            JSONValidator._check_tagged_finding_list(
                items=item.get("common_with_patient", []),
                warnings=warnings,
                errors=errors,
                where=f"comparison_summary[{i}].common_with_patient",
                require_tags=False,
            )
            JSONValidator._check_tagged_finding_list(
                items=item.get("differences_from_patient", []),
                warnings=warnings,
                errors=errors,
                where=f"comparison_summary[{i}].differences_from_patient",
                require_tags=False,
            )

            # Preserve additional checks for prompt/meta phrasing in differences.
            diffs = item.get("differences_from_patient", [])
            if isinstance(diffs, list):
                for diff in diffs:
                    if not isinstance(diff, str):
                        continue
                    diff_low = diff.strip().lower()
                    if diff_low.startswith("no mention of"):
                        warnings.append(f'"No mention of" phrasing in differences case {i}')
                    if diff_low in {"no findings.", "no findings", "none.", "none"}:
                        warnings.append(f'Meta "no findings/none" in differences case {i}')

        # Validate final confidence values when present.
        fd = data.get("final_diagnosis_assessment", {})
        conf = fd.get("confidence") if isinstance(fd, dict) else None
        if isinstance(conf, str) and conf.strip().lower() not in JSONValidator.ALLOWED_CONFIDENCE:
            warnings.append(f"Unexpected confidence value: {conf}")
        if not isinstance(fd, dict):
            errors.append("final_diagnosis_assessment must be a dict")

        logic_ok = len(errors) == 0
        return {"ok": logic_ok, "errors": errors, "warnings": warnings, "logic_ok": logic_ok}
