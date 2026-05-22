# llm/prompts/builders.py

from .schemas import PLAN_OUTPUT_SCHEMA_TEXT
from .utils import format_value, sort_image_roles, to_pretty_json


def build_plan_prompt(
    query: str,
    uploaded_types: list[str],
    wiki_context: str,
    rag_context: str,
    available_models: list[dict] | None = None,
    mode: str = "auto",
) -> str:
    uploaded_str = ", ".join(uploaded_types) if uploaded_types else "없음"
    available_models = available_models or []
    available_models_str = to_pretty_json(available_models)

    return f"""## User Request
{query}

## Selected Mode
{mode}

## Uploaded Data Types
{uploaded_str}

## Available Registered Models
```json
{available_models_str}
```

## Wiki Reference
{wiki_context if wiki_context else "관련 Wiki 정보 없음"}

## RAG Search Results
{rag_context if rag_context else "관련 논문/QA 없음"}

## Instructions
Based on the information above, determine which of the following request types applies:
1. **execution** — requires running one or more registered specialized AI models
2. **knowledge** — a clinical/medical knowledge or concept question
3. **general** — broad or comprehensive analysis ("전반적으로", "종합적으로") without a specific registered model

Mode-specific routing rules:
- If mode is "auto": classify freely using the criteria above. If uploaded data exists but no matching registered model is found, prefer "general" over "execution".
- If mode is "prediction": always return "execution". Select only from the models listed in "Available Registered Models". If no suitable model is found, set missing_inputs to explain why.
- If mode is "clinical": always return "knowledge".
- If mode is "general": always return "general".

Critical constraint: You must only use model names, departments, and project names that appear exactly in the "Available Registered Models" list above. Do not invent or guess model names.

Return only valid JSON matching one of the schemas below. No markdown fences, no extra text.

{PLAN_OUTPUT_SCHEMA_TEXT}"""


def build_clinical_prompt(query: str, wiki_context: str, rag_context: str) -> str:
    return f"""## Clinical Question
{query}

## Wiki Reference
{wiki_context if wiki_context else "관련 Wiki 정보 없음"}

## Reference Literature / QA
{rag_context if rag_context else "관련 논문/QA 없음"}

## Instructions
Answer the clinical question based on the Wiki reference and literature/QA evidence above.
- Explain supporting evidence specifically when available; clearly state uncertainty when not.
- Do not mention AI model routing, execution logic, or system implementation details.
- Write in natural Korean."""


def build_interpret_prompt(
    query: str,
    task: dict,
    execution_context: dict,
    step_results: list[dict],
    wiki_context: str,
    image_roles: list[str] | None = None,
) -> str:
    task_str = f"{task.get('department', '')} / {task.get('project', '')}"
    mode = execution_context.get("mode", "")
    plan = execution_context.get("plan", {})
    plan_str = to_pretty_json(plan) if plan else "없음"

    results_str = ""
    for r in step_results:
        results_str += f"\n### Step {r.get('step', '?')} - {r.get('model', '알 수 없음')}\n"
        results_str += f"- result_type: {r.get('result_type', '')}\n"
        results_str += f"- predictions: {format_value(r.get('predictions'))}\n"
        if r.get("model_output"):
            results_str += f"- model_output: {format_value(r['model_output'])}\n"
        for img in r.get("images", []):
            if isinstance(img, str):
                results_str += "- 이미지 첨부: role=output_image (VLM 참조)\n"
            else:
                results_str += f"- 이미지 첨부: role={img.get('role', '')} (VLM 참조)\n"

    image_roles = image_roles or []
    image_instruction = ""
    if image_roles:
        ordered_roles = sort_image_roles(image_roles)
        roles_str = "\n".join(f"  - {r}" for r in ordered_roles)
        image_instruction = f"""
## Available Images
{roles_str}

## Image Insertion Rules
Insert `[IMG:role]` markers only where the image genuinely aids understanding of the finding being described:
- Only insert an image when the surrounding text directly references or is explained by that image. Do not insert images just because they exist.
- Each role should appear at most once. Only repeat a role if a clearly separate finding in a different section cannot be understood without it.
- Place each marker on its own line, immediately after the sentence it supports.
- Use the role name exactly as provided — do not rename or abbreviate.
- Do not use HTML tags (`<figure>`, `<img>`); use only `[IMG:role]` tokens.
"""

    return f"""## User Request
{query}

## Execution Target
{task_str} (mode: {mode})

## Execution Plan
{plan_str}

## Model Inference Results
{results_str}

## Wiki Reference
{wiki_context if wiki_context else "관련 Wiki 정보 없음"}
{image_instruction}
## Instructions
Interpret the inference results and images above, and write a clinical report for medical professionals.
- Explain the clinical significance of each step's result.
- Use probability values, model outputs (ROI, Grad-CAM, etc.) as the basis for findings.
- Do not overstate the AI result as a definitive diagnosis; include uncertainty and limitations.
- Structured data (classifications, per-sample results, interval aggregates, probability comparisons) should be presented as a markdown table first, followed by an interpretation paragraph.
- Only insert `[IMG:role]` tokens where the image directly supports the finding described in that sentence. Do not insert images out of completeness.
- Prefer inserting each role once; repeat only if a separate section genuinely cannot be understood without seeing the image again.
- Do not include HTML in the response.
- Include recommended follow-up examinations or clinical considerations.
Write in natural Korean."""


def build_general_prompt(query: str, csv_data: list[dict] | None = None) -> str:
    csv_data = csv_data or []
    csv_section = ""
    if csv_data:
        csv_section = f"\n## Numeric Data (CSV)\n```json\n{to_pretty_json(csv_data[:50])}\n```"

    return f"""## User Request
{query}{csv_section}

## Instructions
Analyze all attached data (images, numeric data, etc.) comprehensively and answer the user's request.
Integrate image findings, abnormal numeric values, and clinical relevance into a concise first-pass screening summary.
Write in natural Korean."""
