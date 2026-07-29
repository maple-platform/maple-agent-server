# llm/prompts/builders.py

from .schemas import BOARD_READER_OUTPUT_SCHEMA_TEXT, PLAN_OUTPUT_SCHEMA_TEXT
from .utils import format_metadata, format_value, sort_image_roles, to_pretty_json


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

    # 원본 스캔 컨텍스트 (general 종합판독) — execution_context.attachments(_meta)
    orig_source = execution_context.get("attachments") or execution_context.get("attachments_meta") or []
    orig_section = ""
    if orig_source:
        orig_lines = []
        total_imgs = 0
        for idx, att in enumerate(orig_source, 1):
            fname = att.get("filename") or f"attachment_{idx}"
            atype = att.get("type") or "unknown"
            meta = att.get("metadata") or {}
            line = f"- {fname} ({atype})"
            if meta:
                line += f": {format_metadata(meta)}"
            orig_lines.append(line)
            total_imgs += len(att.get("images") or [])
        note = f"\n원본 스캔 이미지 {total_imgs}장이 모델 결과 이미지 뒤에 함께 제공됩니다." if total_imgs else ""
        orig_section = "\n## Original Scan Context\n" + "\n".join(orig_lines) + note + "\n"

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
{orig_section}{image_instruction}
## Instructions
Interpret the inference results and images above, and write a clinical report for medical professionals.
- Use the original scan context (modality, body part, age, sex, etc.) as clinical grounding when interpreting the model results.
- Explain the clinical significance of each step's result.
- Use probability values, model outputs (ROI, Grad-CAM, etc.) as the basis for findings.
- Do not overstate the AI result as a definitive diagnosis; include uncertainty and limitations.
- Structured data (classifications, per-sample results, interval aggregates, probability comparisons) should be presented as a markdown table first, followed by an interpretation paragraph.
- Only insert `[IMG:role]` tokens where the image directly supports the finding described in that sentence. Do not insert images out of completeness.
- Prefer inserting each role once; repeat only if a separate section genuinely cannot be understood without seeing the image again.
- Do not include HTML in the response.
- Include recommended follow-up examinations or clinical considerations.
Write in natural Korean."""


# ── Clinical Board builders (v4) ───────────────────────────────────────────────

def _format_step_results(step_results: list[dict]) -> str:
    """step_results를 프롬프트용 텍스트로 조립 (build_interpret_prompt에서 이식)."""
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
    return results_str


def _format_original_scan_context(execution_context: dict) -> str:
    """원본 스캔 컨텍스트 섹션 (build_interpret_prompt에서 이식)."""
    orig_source = execution_context.get("attachments") or execution_context.get("attachments_meta") or []
    if not orig_source:
        return ""
    orig_lines = []
    total_imgs = 0
    for idx, att in enumerate(orig_source, 1):
        fname = att.get("filename") or f"attachment_{idx}"
        atype = att.get("type") or "unknown"
        meta = att.get("metadata") or {}
        line = f"- {fname} ({atype})"
        if meta:
            line += f": {format_metadata(meta)}"
        orig_lines.append(line)
        total_imgs += len(att.get("images") or [])
    note = f"\n원본 스캔 이미지 {total_imgs}장이 모델 결과 이미지 뒤에 함께 제공됩니다." if total_imgs else ""
    return "\n## Original Scan Context\n" + "\n".join(orig_lines) + note + "\n"


def _image_instruction(image_roles: list[str] | None) -> str:
    """[IMG:role] 삽입 규칙 섹션 (build_interpret_prompt에서 이식)."""
    image_roles = image_roles or []
    if not image_roles:
        return ""
    ordered_roles = sort_image_roles(image_roles)
    roles_str = "\n".join(f"  - {r}" for r in ordered_roles)
    return f"""
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


def build_reader_prompt(
    query: str,
    task: dict,
    execution_context: dict,
    step_results: list[dict],
    wiki_context: str,
    image_roles: list[str] | None = None,
) -> str:
    """Reader 프롬프트 — 기존 interpret를 UI용 3필드와 검증용 claims로 구조화."""
    task_str = f"{task.get('department', '')} / {task.get('project', '')}"
    mode = execution_context.get("mode", "")
    plan = execution_context.get("plan", {})
    plan_str = to_pretty_json(plan) if plan else "없음"

    results_str = _format_step_results(step_results)
    orig_section = _format_original_scan_context(execution_context)
    image_instruction = _image_instruction(image_roles)

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
{orig_section}{image_instruction}
## Instructions
Interpret the inference results and images above into three Korean, human-facing fields and a structured claim breakdown.

For the human-facing fields:
- `finding`: observed findings and structured model results. Put probability tables here when useful.
- `interpretation`: clinical meaning, uncertainty, and relevant differential considerations.
- `recommendation`: recommended follow-up, clinical correlation, and limitations.
- Use the original scan context (modality, body part, age, sex, etc.) as clinical grounding.
- Explain the clinical significance of each step's result, using probability values and model outputs (ROI, Grad-CAM, etc.) as the basis for findings.
- Do not overstate the AI result as a definitive diagnosis; include uncertainty and limitations.
- Insert `[IMG:role]` only inside the field whose surrounding text directly discusses that image, each role preferably once. Do not include HTML.

For the structured fields:
- `claims`: a list of the discrete clinical claims your opinion makes. Each claim = {{"label": "<short finding/condition name>", "text": "<the claim in one sentence>"}}. Use the SAME label wording that appears in the model predictions when a claim corresponds to a predicted class, so the claim can be matched back to its score.
- `sentence_map`: map each claim to its supporting sentence and field.

{BOARD_READER_OUTPUT_SCHEMA_TEXT}"""


def build_challenger_prompt(
    query: str,
    task: dict,
    step_results: list[dict],
    wiki_context: str,
    alt_model_output: dict | None = None,
    ddx_context: str = "",
) -> str:
    """Challenger 프롬프트 — 두 변형.
    - alt_model_output 있으면: 대체 모델 output 기반 (블라인드)
    - 없으면: 블라인드 동일모델 + RAG DDx 체크리스트

    두 변형 모두 Reader의 구조화 판독/claims는 제공하지 않는다(블라인드 원칙)."""
    task_str = f"{task.get('department', '')} / {task.get('project', '')}"
    results_str = _format_step_results(step_results)

    if alt_model_output is not None:
        source_section = f"""## Alternative Model Output
An alternative model targeting the same clinical question was run. Base your differential primarily on this independent output:
```json
{to_pretty_json(alt_model_output)}
```

## Primary Model Inference Results (for reference only)
{results_str}"""
        source_note = "You have an independent alternative model's output. Compare it against the primary model results and surface any divergent or additional findings."
    else:
        source_section = f"""## Model Inference Results
{results_str}

## Differential Diagnosis Checklist (retrieved)
{ddx_context if ddx_context else "관련 감별진단 참고 정보 없음"}"""
        source_note = "No alternative model is available. Use the differential diagnosis checklist above to systematically consider alternatives the primary reading might miss."

    return f"""## User Request
{query}

## Execution Target
{task_str}

{source_section}

## Wiki Reference
{wiki_context if wiki_context else "관련 Wiki 정보 없음"}

## Instructions
You are producing an INDEPENDENT differential. You have NOT been shown the primary reading — do not assume what it concluded. {source_note}
- List plausible alternative or additional findings, each with a one-line rationale grounded in the outputs/images above.
- Include findings that would be clinically important not to miss, even at lower probability.

Return only valid JSON (no markdown fences) in exactly this shape:
{{
  "differential": [{{"label": "<finding/condition name>", "rationale": "<one-line reason>"}}]
}}"""


def build_evidence_prompt(
    query: str,
    reader_claims: list[dict],
    challenger_differential: list[dict],
    rag_context: str,
) -> str:
    """Evidence 프롬프트 — Reader.claims ∪ Challenger.differential을 주장 단위로 RAG 대조."""
    reader_str = to_pretty_json(reader_claims) if reader_claims else "[]"
    challenger_str = to_pretty_json(challenger_differential) if challenger_differential else "[]"

    return f"""## User Request
{query}

## Reader Claims
```json
{reader_str}
```

## Challenger Differential
```json
{challenger_str}
```

## Retrieved Clinical Evidence
{rag_context if rag_context else "관련 근거 없음"}

## Instructions
For each claim (from both the Reader claims and the Challenger differential), assess it against the retrieved clinical evidence.
- Classify support as "supported", "neutral", or "unsupported".
- A claim is "unsupported" only when the retrieved evidence provides no backing (not merely because evidence is silent — use "neutral" for silence).
- Do not invent evidence beyond the retrieved context.

Return only valid JSON (no markdown fences) in exactly this shape:
{{
  "evidence_map": [{{"claim": "<claim label or text>", "support": "supported|neutral|unsupported", "note": "<short justification>"}}],
  "unsupported_claims": ["<verbatim claim label or text>", ...]
}}"""


def build_guardian_prompt(
    query: str,
    task: dict,
    step_results: list[dict],
    reader_out: dict,
    challenger_out: dict,
    evidence_out: dict,
    guardian_rag_context: str,
) -> str:
    """Guardian 프롬프트 — 자체 RAG 재조회 + 최종 veto."""
    task_str = f"{task.get('department', '')} / {task.get('project', '')}"
    safe_step_results = []
    for step in step_results:
        safe_step = {k: v for k, v in step.items() if k != "images"}
        safe_step["image_roles"] = [
            image.get("role", "") if isinstance(image, dict) else "output_image"
            for image in step.get("images", [])
        ]
        safe_step_results.append(safe_step)
    case_str = to_pretty_json({
        "step_results": safe_step_results,
        "reader": reader_out,
        "challenger": challenger_out,
        "evidence": evidence_out,
    })

    return f"""## User Request
{query}

## Execution Target
{task_str}

## Complete Board Case
```json
{case_str}
```

## Guardian Re-retrieved Context
{guardian_rag_context if guardian_rag_context else "관련 근거 없음"}

## Instructions
As the final safety net, independently re-review this case.
- Consider whether any don't-miss / critical finding may be present or inadequately addressed, grounded in the re-retrieved clinical context.
- Consider whether the unsupported claims from the Evidence agent create patient-safety risk.
- Set veto to true only when the interpretation should NOT be auto-confirmed without human review.
- Assign risk_tier as low, moderate, high, or critical. High/critical require a concrete rationale grounded in the supplied case or retrieved evidence.
- Be conservative: when patient safety is in doubt, flag it.

Return only valid JSON (no markdown fences) in exactly this shape:
{{
  "veto": false,
  "risk_tier": "low|moderate|high|critical",
  "flags": ["<safety concern>", ...],
  "rationale": "<evidence-grounded short reason>",
  "evidence_refs": ["<supporting reference from retrieved context>", ...]
}}"""


def build_intent_prompt(query: str, uploaded_types: list[str] | None = None) -> str:
    uploaded_str = ", ".join(uploaded_types) if uploaded_types else "없음"
    return f"""## User Request
{query}

## Uploaded File Types
{uploaded_str}

## Instructions
Analyze the request and extract the clinical intent for specialized-model retrieval.
Return only this JSON (values in English):
{{
  "body_part": "target body part or anatomy, or empty string",
  "disease_group": "disease/condition group, or empty string",
  "modality": "imaging modality if identifiable (MR, CT, X-ray, ...), or empty string",
  "search_query": "a concise English phrase describing the analysis task, optimized for semantic model search"
}}"""


def build_model_select_prompt(query: str, intent: dict, candidates: list[dict]) -> str:
    """후보 모델 중 쿼리에 적절한 것을 LLM이 고르게 하는 프롬프트.
    candidates: [{"metadata": {...}, "text": "<doc_text>"}]"""
    lines = []
    for i, c in enumerate(candidates, 1):
        m = c.get("metadata", {})
        # doc_text 전문을 보여준다 — 질환(disease) 목록이 잘려 핵심 키워드가 누락되지 않도록.
        info = (c.get("text") or "").strip().replace("\n", " ")[:1200]
        lines.append(
            f"{i}. model_name: {m.get('model_name', '')}\n"
            f"   department: {m.get('department', '')} | project: {m.get('project', '')}\n"
            f"   task_type: {m.get('task_type', '')} | required_data: {m.get('required_data', '')}\n"
            f"   info: {info}"
        )
    candidates_str = "\n".join(lines) if lines else "(no candidates)"

    return f"""## User Request
{query}

## Analyzed Intent
{to_pretty_json(intent)}

## Candidate Models
{candidates_str}

## Instructions
Select the models appropriate to fulfill the user's request, using the request, the analyzed intent, and each candidate's info/task.
- A model is relevant if it can detect, classify, or otherwise identify the target condition — including when the condition appears among the findings listed in its info (질환/disease). Do NOT exclude a model just because its task_type wording (e.g. "classification") differs from the request wording (e.g. "detection").
- If several relevant models exist for the same condition (e.g. a detection model AND a classification model that covers it), include ALL of them.
- If none are appropriate, return an empty list.
- Use model_name values exactly as listed. Do not invent models.

Return only this JSON, no markdown fences:
{{"selected_models": ["model_name", ...]}}"""


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


def build_general_prompt_from_attachments(
    query: str,
    attachments: list[dict],
) -> tuple[str, list[str]]:
    """정규화된 attachments[]로 general 프롬프트를 조립.
    반환: (prompt, VLM에 순서대로 넣을 이미지 목록)

    이미지가 파일당 여러 장(예: 3면 슬라이스) 올 수 있으므로,
    각 이미지를 "Attachment i image j" 라벨로 명시하고 VLM 입력 순서와 일치시킨다.
    """
    sections: list[str] = []
    vlm_images: list[str] = []
    image_manifest: list[str] = []

    for i, att in enumerate(attachments, 1):
        atype = att.get("type") or "unknown"
        fname = att.get("filename") or f"attachment_{i}"
        lines = [f"### Attachment {i}: {fname} ({atype})"]

        meta = att.get("metadata") or {}
        if meta:
            lines.append(f"- Metadata: {format_metadata(meta)}")

        text = (att.get("text") or "").strip()
        if text:
            lines.append(f"- Extracted text: {text}")

        tabular = att.get("tabular")
        if tabular:
            lines.append(f"- Tabular data:\n```json\n{to_pretty_json(tabular[:50])}\n```")

        imgs = att.get("images") or []
        if imgs:
            labels = []
            for j, img in enumerate(imgs, 1):
                vlm_images.append(img)
                label = f"Attachment {i} image {j}"
                labels.append(label)
                image_manifest.append(label)
            lines.append(f"- Attached images: {', '.join(labels)}")

        sections.append("\n".join(lines))

    attachments_str = "\n\n".join(sections) if sections else "없음"

    manifest_str = ""
    if image_manifest:
        ordered = "\n".join(f"{k}. {label}" for k, label in enumerate(image_manifest, 1))
        manifest_str = (
            "\n\n## Image Order\n"
            "The images are provided to you in exactly this order:\n" + ordered
        )

    prompt = f"""## User Request
{query}

## Attachments
{attachments_str}{manifest_str}

## Instructions
Analyze all attached data (images, extracted text, metadata, tabular values) comprehensively and answer the user's request.
- Use each attachment's metadata (modality, body part, age, sex, etc.) as clinical context when interpreting its images.
- Refer to images by their attachment and sequence when relevant.
Integrate image findings, abnormal numeric values, and clinical relevance into a concise first-pass screening summary.
Write in natural Korean."""
    return prompt, vlm_images
