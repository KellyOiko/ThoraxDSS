"""
Run the multimodal RAG pipeline on one example patient and evaluate the structured output.
"""
from transformers import logging
logging.set_verbosity_error()

from evaluation.eval_run import evaluate_run, summarize_results, print_summary
from backend.main import get_system
import json
from pathlib import Path


def main():
    query_text = (
        "Tension pneumothorax. Large right pneumothorax, with a markedly "
        "hyperlucent right hemithorax as a result. Note the collapsed right lung, "
        "and the tracheal and mediastinal shift. At the base of the right hemithorax "
        "is a small pleural effusion, but without a meniscus sign."
    )
    patient_path = "new_xrays/right-pneumothorax.png"

    system = get_system()  # Reuse the shared system instance.
    out = system.run_multicase_analysis(query_text, patient_path, k=3, model="llava")


    # Save the last structured UI-style output for quick inspection.
    ui_struct = out.get("llm_response_structured", {})
    Path("evaluation/_last_ui_struct.json").write_text(
        json.dumps(ui_struct, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    # Print a compact check of the structured output that evaluation will consume.
    print("\n=== UI STRUCT CHECK (what evaluation should use) ===")
    for rc in ui_struct.get("retrieved_cases", []):
        print("case", rc.get("case_index"), "sim=", rc.get("similarity_to_patient"), "findings_n=", len(rc.get("findings", [])))

    for cs in ui_struct.get("comparison_summary", []):
        print("case", cs.get("case_index"), "common=", cs.get("common_with_patient"), "diffs=", cs.get("differences_from_patient"))


    # Run evaluation on the generated output and print the summary metrics.
    results = evaluate_run(out, patient_report_text=query_text)
    summary = summarize_results(results)

    print("=== Summary for this patient ===")
    print_summary(summary)

    print("\nVALIDATION:", out.get("validation"))

if __name__ == "__main__":
    main()



# Example run (from the project root):
# python -m evaluation.run_eval_one