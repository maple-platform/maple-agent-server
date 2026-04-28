SYSTEM_PLAN = """당신은 MARS AI 플랫폼의 AI Agent입니다.
사용자의 요청을 분석하여 적절한 AI 모델 실행 계획을 수립하거나 의학 지식을 제공합니다.
항상 한국어로 응답하세요."""

SYSTEM_CLINICAL = """당신은 MARS AI 플랫폼의 임상 의학 전문 AI Agent입니다.
PubMedQA, MedMCQA 논문 데이터와 Wiki 지식을 바탕으로 근거 기반 임상 답변을 제공합니다.
항상 한국어로 응답하고, 참고한 근거를 명확히 설명하세요."""

SYSTEM_INTERPRET = """당신은 MARS AI 플랫폼의 임상 해석 전문 AI Agent입니다.
AI 모델의 추론 결과를 임상적 맥락에서 해석하여 의미 있는 설명을 제공합니다.
항상 한국어로 응답하고, 확률 수치와 임상적 의의를 명확히 설명하세요."""

SYSTEM_GENERAL = """당신은 MARS AI 플랫폼의 범용 의료 영상 분석 AI Agent입니다.
첨부된 이미지와 사용자 질문을 바탕으로 시각적·의학적 분석 결과를 제공합니다.
항상 한국어로 응답하세요."""


def build_plan_prompt(query: str, uploaded_types: list[str], wiki_context: str, rag_context: str) -> str:
    uploaded_str = ", ".join(uploaded_types) if uploaded_types else "없음"
    return f"""## 사용자 요청
{query}

## 업로드된 데이터 타입
{uploaded_str}

## Wiki 참고 정보
{wiki_context if wiki_context else "관련 Wiki 정보 없음"}

## RAG 검색 결과
{rag_context if rag_context else "관련 논문/QA 없음"}

## 지시
위 정보를 바탕으로 요청이 다음 중 어느 유형인지 판단하세요:
1. **모델 실행 요청**: 등록된 특화 AI 모델 실행이 필요한 경우
2. **임상 지식 질문**: 의학 지식, 개념, 질환 설명 요청
3. **범용 종합 분석**: "전반적으로", "종합적으로", "전체적으로" 등 특화 모델 없이 데이터를 전반 분석하는 요청

모델 실행 요청이면:
{{"type": "execution", "steps": [{{"step": 1, "model": "모델명", "department": "진료과", "project": "프로젝트명"}}], "missing_inputs": [], "message": "설명"}}

임상 지식 질문이면:
{{"type": "knowledge", "message": "답변 내용"}}

범용 종합 분석이면:
{{"type": "general", "message": "범용 분석을 수행합니다."}}

JSON만 반환하세요."""


def build_clinical_prompt(query: str, wiki_context: str, rag_context: str) -> str:
    return f"""## 임상 질문
{query}

## 참고 논문/QA
{rag_context if rag_context else "관련 논문/QA 없음"}

## 지시
위 논문/QA 근거를 바탕으로 임상 질문에 답변하세요.
- 의학적 근거가 있으면 구체적으로 설명하고, 불확실한 부분은 명시하세요.
- AI 모델이나 시스템 관련 내용은 답변에 포함하지 마세요.
- 자연스러운 한국어 문장으로 답변하세요."""


def build_interpret_prompt(
    query: str,
    task: dict,
    execution_context: dict,
    step_results: list[dict],
    wiki_context: str,
    image_roles: list[str] = [],
) -> str:
    import json
    import re

    def format_value(value) -> str:
        if value in (None, "", [], {}):
            return "없음"
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    def sort_image_roles(roles: list[str]) -> list[str]:
        def sort_key(role: str):
            m = re.fullmatch(r"(.+?)_(\d+)", role)
            if m:
                return (m.group(1), int(m.group(2)))
            return (role, -1)
        return sorted(roles, key=sort_key)

    task_str = f"{task.get('department', '')} / {task.get('project', '')}"
    mode = execution_context.get("mode", "")
    plan = execution_context.get("plan", {})
    plan_str = json.dumps(plan, ensure_ascii=False) if plan else "없음"

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

    # 이미지 삽입 지시 블록 (이미지가 있을 때만 추가)
    image_instruction = ""
    if image_roles:
        ordered_roles = sort_image_roles(image_roles)
        roles_str = "\n".join(f"  - {r}" for r in ordered_roles)
        image_instruction = f"""
## 사용 가능한 이미지 목록
{roles_str}

## 이미지 삽입 규칙
보고서 작성 시, 위 이미지 목록 중 해당 설명과 가장 관련 있는 위치에 아래 형식의 마커를 삽입하세요.
형식: [IMG:role명]
- 각 이미지는 관련 설명 문단이 끝난 직후(단락 사이 빈 줄)에 삽입
- 이미지 마커는 텍스트 줄과 별도 줄로 분리
- role 이름은 전달된 값을 그대로 사용하세요. 예: `bbox_overlay`, `seg3d_1`, `seg3d_2`
- 이미지가 여러 장이면 대표 몇 장만 임의로 고르지 말고, 설명의 근거가 되는 이미지 범위를 빠짐없이 모두 포함하세요.
- 이미지 토큰 배치 순서는 인덱스 순서가 아니라 해석 맥락을 우선하세요.
- 전체 결과를 요약하는 이미지는 도입부에, 세부 분석용 이미지는 해당 내용을 설명하는 문장 바로 뒤에 배치하세요.
- 같은 이미지가 여러 구조/부위/소견의 근거가 되면 중복 삽입을 허용하고, 필요한 모든 섹션에 다시 포함하세요.
- 슬라이스를 구조별로 배타적으로 한 번만 배정하지 말고, 각 라벨이 보이는지 여부를 기준으로 독립적으로 판단하세요.
- HTML 태그(`<figure>`, `<img>`)는 절대 사용하지 말고 `[IMG:role]` 토큰만 사용하세요.
"""

    return f"""## 사용자 요청
{query}

## 실행 대상
{task_str} (모드: {mode})

## 실행 계획
{plan_str}

## 모델 추론 결과
{results_str}

## Wiki 참고 정보
{wiki_context if wiki_context else "관련 Wiki 정보 없음"}
{image_instruction}
## 지시
위 추론 결과와 이미지를 종합하여 임상적으로 해석하고, 의사가 이해할 수 있는 보고서를 작성하세요.
- 각 단계별 결과의 임상적 의의를 설명하세요.
- 확률 수치와 모델 출력값(ROI, GradCAM 등)을 근거로 소견을 기술하세요.
- 첨부 이미지가 있으면 해당 이미지를 언급하는 문장 바로 뒤에 `[IMG:role]` 토큰을 삽입하세요.
- role 이름을 임의로 바꾸거나 축약하지 말고, 전달된 role 이름을 그대로 사용하세요.
- 이미지 토큰은 숫자 인덱스 순서보다 설명 흐름과 임상적 맥락에 맞춰 자연스럽게 배치하세요.
- 구조별, 부위별, 단계별, 시점별, 슬라이스별 설명이 있으면 그 설명에 대응하는 이미지 role도 같은 순서로 모두 포함하세요.
- 설명한 시각적 근거의 범위와 첨부한 `[IMG:role]` 토큰 범위가 항상 일치해야 합니다.
- 구조별 설명 섹션은 서로 배타적이면 안 됩니다.
- 같은 슬라이스에 여러 구조나 라벨이 동시에 보이면, 그 슬라이스 이미지는 관련된 모든 구조 설명 섹션에 중복 포함되어야 합니다.
- 슬라이스를 구조별로 나눠 한 번만 배정하는 partition 방식은 사용하지 말고, 각 라벨 기준으로 독립적으로 선택하세요.
- 분류 결과, 샘플별 결과, 구간별 집계, 확률/빈도 비교처럼 구조화 가능한 정보는 가능하면 마크다운 테이블로 먼저 정리하세요.
- 특히 prediction/classification 결과는 텍스트만 길게 풀어쓰기보다 핵심 결과를 표로 요약한 뒤 해석 문단을 이어서 작성하세요.
- 표가 더 읽기 쉬운 경우에는 표를 우선 사용하고, 그 아래에 임상적 의미를 자연스럽게 설명하세요.
- 응답에는 HTML을 넣지 말고 `[IMG:role]` 토큰만 사용하세요.
- 추가 검사 또는 임상 권고사항을 포함하세요.
응답은 자연스러운 한국어 문장으로 작성하세요."""


def build_general_prompt(query: str, csv_data: list[dict] = []) -> str:
    csv_section = ""
    if csv_data:
        import json
        csv_section = f"\n## 수치 데이터 (CSV)\n```json\n{json.dumps(csv_data[:50], ensure_ascii=False, indent=2)}\n```"

    return f"""## 사용자 요청
{query}{csv_section}

## 지시
첨부된 모든 데이터(이미지, 수치 데이터 등)를 종합적으로 분석하고 사용자 요청에 답변하세요.
이미지 소견, 수치 이상 여부, 임상적 의의를 통합하여 1차 스크리닝 소견으로 작성하세요."""
