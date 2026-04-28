import json
import logging
import re
import httpx

from services import wiki_service
from services.rag_service import retrieve
from rag.retriever import search_models
from llm import client as llm_client
from llm.prompts import (
    SYSTEM_PLAN,
    SYSTEM_CLINICAL,
    SYSTEM_INTERPRET,
    SYSTEM_GENERAL,
    build_plan_prompt,
    build_clinical_prompt,
    build_interpret_prompt,
    build_general_prompt,
)

logger = logging.getLogger("mars-ai-agent")


async def plan(
    query: str,
    uploaded_types: list[str],
    history: list[dict],
    mode: str = "auto",
    images: list[str] = [],
    csv_data: list[dict] = [],
) -> dict:
    if mode == "clinical":
        return await _clinical(query)

    if mode == "prediction":
        return await _prediction(query, uploaded_types)

    if mode == "general":
        return await _general(query, images, csv_data)

    # mode == "auto": 데이터가 있으면 general 우선 분기
    if images or csv_data:
        return await _general(query, images, csv_data)

    return await _auto(query, uploaded_types)


# ── 파일 타입 정규화 ───────────────────────────────────────────────────────────

# 의료 영상 파일 확장자 → 표준 포맷명 매핑
_EXT_NORMALIZE: dict[str, str] = {
    # DICOM
    "dcm": "dicom",
    "dicom": "dicom",
    "ima": "dicom",         # Siemens DICOM
    # NIfTI (Neuroimaging Informatics Technology Initiative)
    "nii": "nifti",
    "nii.gz": "nifti",      # gzip 압축 NIfTI (복합 확장자)
    # Analyze (NIfTI 이전 포맷, 대부분의 파이프라인에서 호환)
    "hdr": "analyze",
    "img": "analyze",
    # FreeSurfer MGH/MGZ
    "mgh": "mgh",
    "mgz": "mgh",           # gzip 압축 MGH
    # MINC (Montreal Neurological Institute)
    "mnc": "minc",
    # NRRD (Nearly Raw Raster Data, ITK 표준)
    "nrrd": "nrrd",
    "nhdr": "nrrd",
    "seg.nrrd": "nrrd",
    # MetaImage (ITK/SimpleITK)
    "mha": "metaimage",
    "mhd": "metaimage",
    # Whole Slide Imaging (병리학)
    "svs": "wsi",           # Aperio
    "ndpi": "wsi",          # Hamamatsu
    "scn": "wsi",           # Leica
    "mrxs": "wsi",          # MIRAX/3DHistech
    "ome.tif": "wsi",       # OME-TIFF
    # 표준 이미지 (안저/피부과/병리 등)
    "png": "image",
    "jpg": "image",
    "jpeg": "image",
    "bmp": "image",
    "tif": "tiff",
    "tiff": "tiff",
    # 임상 데이터
    "csv": "csv",
    "xlsx": "excel",
    "xls": "excel",
    "json": "json",
    # ML/데이터 포맷
    "npy": "numpy",
    "npz": "numpy",
    "h5": "hdf5",
    "hdf5": "hdf5",
}

# NIfTI 포맷과 호환되는 원시 확장자 목록
# - 백엔드가 "nii.gz" 파일을 처리할 때 마지막 확장자인 "gz"만 전송하는 경우 대응
# - 단, "gz" 단독은 모호함 (mnc.gz, tar.gz 등 존재) → nifti 요구 모델일 때만 조건부 허용
_NIFTI_COMPAT_RAW = {"gz", "nii", "nii.gz", "nifti"}


def _normalize_types(uploaded_types: list[str]) -> list[str]:
    """업로드된 파일 확장자를 표준 포맷명으로 정규화.
    복합 확장자(nii.gz 등)는 그대로 우선 처리하고, 단순 확장자는 매핑 적용.
    "gz" 단독은 포맷 불명확으로 정규화하지 않고 원본 유지."""
    normalized = []
    for t in uploaded_types:
        t_lower = t.lower().strip()
        mapped = _EXT_NORMALIZE.get(t_lower, t_lower)
        if mapped not in normalized:
            normalized.append(mapped)
    return normalized


def _types_match(required_data: list[str], uploaded_types: list[str]) -> bool:
    """모델의 required_data와 업로드된 파일 타입 매칭 여부 판단.

    NIfTI 조건부 처리:
    - 모델이 "nifti"를 요구하는 경우, "gz" 확장자도 허용
      (백엔드가 .nii.gz 파일의 확장자를 "gz"로만 전송할 수 있으므로)
    """
    normalized = _normalize_types(uploaded_types)
    raw_lower = {t.lower().strip() for t in uploaded_types}

    for req in required_data:
        req_norm = _EXT_NORMALIZE.get(req.lower().strip(), req.lower().strip())
        # 정규화 후 직접 매칭
        if req_norm in normalized:
            return True
        # NIfTI 조건부 매칭: 모델이 nifti 요구 시 gz 등 호환 확장자 허용
        if req_norm == "nifti" and raw_lower & _NIFTI_COMPAT_RAW:
            return True
    return False


# ── mode 핸들러 ───────────────────────────────────────────────────────────────

async def _clinical(query: str) -> dict:
    """RAG + Wiki 즉시 검색 → 임상 지식 답변"""
    wiki_context = wiki_service.search_wiki(query)
    model_results, knowledge_results, rag_context = await retrieve(query)

    prompt = build_clinical_prompt(query, wiki_context, rag_context)
    message = await llm_client.generate(prompt, system=SYSTEM_CLINICAL)

    sources = []
    for r in knowledge_results:
        sources.append({"source": "pubmedqa", "text": r["text"][:80], "score": r["score"]})

    # 관련 특화 모델 추천 문구 생성
    related_models = []
    for r in model_results:
        meta = r.get("metadata", {})
        project = meta.get("project", "")
        model_name = meta.get("model_name", "")
        if project and model_name:
            related_models.append(f"{project} ({model_name})")

    model_suggestion = None
    if related_models:
        model_list = ", ".join(related_models)
        model_suggestion = f"관련 질환에 사용할 수 있는 특화 AI 모델이 있습니다: {model_list}"

    return {
        "query_type": "knowledge",
        "mode": "clinical",
        "message": message,
        "sources": sources,
        "model_suggestion": model_suggestion,
    }


async def _prediction(query: str, uploaded_types: list[str]) -> dict:
    """모델 레지스트리 검색 → required_data 매칭 → 실행 계획 반환"""
    try:
        model_results = search_models(query, n_results=5)
    except Exception:
        model_results = []

    if not model_results:
        return {
            "status": "no_model",
            "mode": "prediction",
            "message": (
                "요청에 맞는 등록된 AI 모델을 찾지 못했습니다. "
                "다른 표현으로 다시 시도하거나 모델 등록 후 사용해 주세요."
            ),
        }

    # 유사도 낮은 모델 제거 (threshold 이하는 관련 없음으로 판단)
    SCORE_THRESHOLD = 0.40
    model_results = [r for r in model_results if r.get("score", 0) >= SCORE_THRESHOLD]

    if not model_results:
        return {
            "status": "no_model",
            "mode": "prediction",
            "message": (
                "요청에 맞는 등록된 AI 모델을 찾지 못했습니다. "
                "다른 표현으로 다시 시도하거나 모델 등록 후 사용해 주세요."
            ),
        }

    # uploaded_types와 required_data 매칭 필터링
    matched = []
    mismatched = []
    for r in model_results:
        meta = r.get("metadata", {})
        model_name = meta.get("model_name", "")
        # ChromaDB metadata에 required_data가 없으면 → Wiki에서 보완
        required_data = _get_required_data(meta, model_name)

        if not required_data:
            # required_data 정보 자체가 없으면 일단 포함
            matched.append(meta)
        elif _types_match(required_data, uploaded_types):
            matched.append(meta)
        else:
            mismatched.append({
                "model": model_name,
                "required": required_data,
                "uploaded": uploaded_types,
            })

    if not matched:
        # 검색은 됐지만 데이터 타입이 안 맞는 경우
        mismatch_desc = "; ".join(
            f"{m['model']} (필요: {m['required']}, 업로드: {m['uploaded']})"
            for m in mismatched
        )
        return {
            "status": "type_mismatch",
            "mode": "prediction",
            "message": (
                "검색된 모델이 요구하는 데이터 타입과 업로드된 파일이 일치하지 않습니다. "
                f"상세: {mismatch_desc}"
            ),
            "mismatched_models": mismatched,
        }

    steps = [
        {
            "step": i + 1,
            "model": m.get("model_name", ""),
            "department": m.get("department", ""),
            "project": m.get("project", ""),
            "task_type": m.get("task_type", ""),
            "result_type": m.get("result_type", ""),
        }
        for i, m in enumerate(matched)
    ]

    return {
        "status": "ready",
        "mode": "prediction",
        "execution_plan": {"steps": steps},
        "missing_inputs": [],
        "message": f"{len(steps)}개 모델 실행 계획이 수립되었습니다.",
    }


def _get_required_data(meta: dict, model_name: str) -> list[str]:
    """ChromaDB metadata에 required_data 없으면 Wiki 페이지에서 파싱"""
    # metadata에 직접 있는 경우 (등록 시 저장했으면)
    if "required_data" in meta:
        val = meta["required_data"]
        if isinstance(val, str):
            return [v.strip() for v in val.split(",") if v.strip()]
        if isinstance(val, list):
            return val

    # Wiki 페이지에서 파싱
    if model_name:
        from services import wiki_service
        page = wiki_service.read_page("models", model_name)
        for line in page.splitlines():
            if "required_data" in line:
                # - **required_data:** [dicom, csv]
                start = line.find("[")
                end = line.find("]")
                if start != -1 and end != -1:
                    return [v.strip() for v in line[start+1:end].split(",") if v.strip()]
    return []


async def _general(query: str, images: list[str], csv_data: list[dict]) -> dict:
    """이미지 + CSV + 텍스트 → VLM 범용 종합 분석"""
    prompt = build_general_prompt(query, csv_data)
    try:
        if images:
            result = await llm_client.generate_with_images(prompt, images, system=SYSTEM_GENERAL)
        else:
            result = await llm_client.generate(prompt, system=SYSTEM_GENERAL)
    except httpx.HTTPStatusError as e:
        detail = e.response.text[:300] if e.response is not None else str(e)
        fallback_prompt = (
            f"{prompt}\n\n"
            "## 시스템 참고\n"
            "첨부 이미지의 VLM 분석 요청이 실패하여 텍스트/수치 데이터만 기반으로 답변하세요. "
            f"Ollama 오류 요약: {detail}"
        )
        result = await llm_client.generate(fallback_prompt, system=SYSTEM_GENERAL)

    return {"query_type": "general", "mode": "general", "message": result}


async def _auto(query: str, uploaded_types: list[str]) -> dict:
    """기존 LLM 기반 자동 분류"""
    wiki_context = wiki_service.search_wiki(query)
    model_results, knowledge_results, rag_context = await retrieve(query)

    prompt = build_plan_prompt(query, uploaded_types, wiki_context, rag_context)
    raw = await llm_client.generate(prompt, system=SYSTEM_PLAN)
    parsed = _parse_json_response(raw)

    if parsed.get("type") == "execution":
        steps = parsed.get("steps", [])
        missing_inputs = parsed.get("missing_inputs", [])
        return {
            "status": "ready" if not missing_inputs else "missing_inputs",
            "mode": "auto",
            "execution_plan": {"steps": steps},
            "missing_inputs": missing_inputs,
            "message": parsed.get("message", "실행 계획이 수립되었습니다."),
        }
    elif parsed.get("type") == "general":
        # auto에서 general로 재분기 (데이터 없이 텍스트만 온 경우)
        return await _general(query, [], [])
    else:
        sources = []
        for r in (model_results + knowledge_results):
            src = "wiki" if r["source"] == "mars_models" else "pubmedqa"
            sources.append({"source": src, "text": r["text"][:80], "score": r["score"]})
        return {
            "query_type": "knowledge",
            "mode": "auto",
            "message": parsed.get("message", raw),
            "sources": sources,
        }


# ── interpret ─────────────────────────────────────────────────────────────────

async def interpret(
    query: str,
    task: dict,
    execution_context: dict,
    step_results: list[dict],
) -> dict:
    step_image_logs: list[dict] = []

    # 1. 관련 모델 Wiki 컨텍스트 수집
    model_names = [r.get("model", "") for r in step_results]
    wiki_ctx_parts = []
    for name in model_names:
        page = wiki_service.read_page("models", name)
        if page:
            wiki_ctx_parts.append(f"### {name}\n{page[:600]}")
    wiki_context = "\n\n".join(wiki_ctx_parts)

    # 2. step_results에서 이미지 추출 (role → data 매핑 유지)
    image_map: dict[str, str] = {}   # role → "data:image/png;base64,..." (원본, 프론트엔드용)
    raw_images: list[str] = []       # 순수 base64 목록 (Ollama VLM용)
    ordered_roles: list[str] = []
    for step in step_results:
        step_roles: list[str] = []
        for img in step.get("images", []):
            if isinstance(img, str):
                role = "output_image"
                data = img
            else:
                role = img.get("role", "")
                data = img.get("data", "")
            if not data:
                continue
            if role:
                image_map[role] = data
                ordered_roles.append(role)
                step_roles.append(role)
            raw = data.split(",", 1)[1] if data.startswith("data:") else data
            raw_images.append(raw)
        step_image_logs.append({
            "step": step.get("step"),
            "model": step.get("model", ""),
            "count": len(step_roles),
            "roles": step_roles,
        })

    image_roles = _unique_preserve_order(ordered_roles)
    logger.info(
        "[interpret] input step image summary=%s",
        step_image_logs,
    )

    # 3. 프롬프트 생성 (이미지 role 목록 포함)
    prompt = build_interpret_prompt(
        query, task, execution_context, step_results, wiki_context,
        image_roles=image_roles,
    )

    # 4. 이미지가 있으면 VLM, 없으면 텍스트 LLM
    try:
        if raw_images:
            interpretation = await llm_client.generate_with_images(prompt, raw_images, system=SYSTEM_INTERPRET)
        else:
            interpretation = await llm_client.generate(prompt, system=SYSTEM_INTERPRET)
    except httpx.HTTPStatusError as e:
        detail = e.response.text[:300] if e.response is not None else str(e)
        fallback_prompt = (
            f"{prompt}\n\n"
            "## 시스템 참고\n"
            "첨부 이미지의 VLM 해석 요청이 실패하여 텍스트 형태의 모델 출력만 기반으로 해석하세요. "
            f"Ollama 오류 요약: {detail}"
        )
        interpretation = await llm_client.generate(fallback_prompt, system=SYSTEM_INTERPRET)
    except httpx.HTTPError as e:
        logger.exception("[interpret] LLM/VLM HTTP error")
        fallback_prompt = (
            f"{prompt}\n\n"
            "## 시스템 참고\n"
            "첨부 이미지의 LLM/VLM 요청 중 네트워크 또는 타임아웃 오류가 발생하여 "
            "텍스트 형태의 모델 출력만 기반으로 해석을 시도합니다. "
            f"Ollama 오류 요약: {type(e).__name__}: {e}"
        )
        try:
            interpretation = await llm_client.generate(fallback_prompt, system=SYSTEM_INTERPRET)
        except Exception:
            logger.exception("[interpret] fallback text generation also failed")
            interpretation = ""
    except Exception as e:
        logger.exception("[interpret] unexpected error during interpretation generation")
        interpretation = ""

    if _needs_interpretation_retry(interpretation):
        retry_prompt = (
            f"{prompt}\n\n"
            "## 재출력 지시\n"
            "방금 응답은 분석 텍스트가 부족했습니다. 이번에는 반드시 충분한 한국어 해석 문장으로 다시 작성하세요.\n"
            "- `## 요약`, `## 주요 영상 소견`, `## 임상적 해석`, `## 권고 또는 한계` 4개 섹션을 모두 포함하세요.\n"
            "- 각 섹션은 최소 2문장 이상 작성하세요.\n"
            "- 이미지 토큰만 나열하지 말고, 실제 해석 문장을 중심으로 작성하세요.\n"
            "- `[IMG:role]` 토큰은 설명 문장 뒤에 배치하되, 본문 분석을 대체하면 안 됩니다."
        )
        try:
            if raw_images:
                interpretation = await llm_client.generate_with_images(retry_prompt, raw_images, system=SYSTEM_INTERPRET)
            else:
                interpretation = await llm_client.generate(retry_prompt, system=SYSTEM_INTERPRET)
        except Exception:
            logger.exception("[interpret] retry generation failed")
            interpretation = ""

    if _needs_interpretation_retry(interpretation):
        logger.warning(
            "[interpret] model output still lacks analysis text after retry; applying server fallback"
        )
        interpretation = _build_interpretation_fallback(
            task=task,
            step_results=step_results,
            image_roles=image_roles,
        )

    interpretation = _ensure_all_image_tokens(interpretation, image_roles)
    token_count = len(re.findall(r"\[IMG:[^\]]+\]", interpretation))
    logger.info(
        "[interpret] output token_count=%s image_keys=%s roles=%s",
        token_count,
        len(image_map),
        image_roles,
    )

    # 5. Wiki에 해석 패턴 누적
    if model_names and model_names[0]:
        wiki_service.append_interpretation(model_names[0], interpretation)
        wiki_service.append_log("interpret", f"{model_names[0]} 결과 해석 완료")

    return {
        "interpretation": interpretation,
        "interpretation_raw": interpretation,
        "images": image_map,
    }


def _sort_image_roles(roles: list[str]) -> list[str]:
    def sort_key(role: str):
        m = re.fullmatch(r"(.+?)_(\d+)", role)
        if m:
            return (m.group(1), int(m.group(2)))
        return (role, -1)
    return sorted(roles, key=sort_key)


def _unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def _ensure_all_image_tokens(text: str, roles: list[str]) -> str:
    if not roles:
        return text

    ordered_roles = _sort_image_roles(_unique_preserve_order(roles))
    missing = [role for role in ordered_roles if f"[IMG:{role}]" not in text]
    if not missing:
        return text

    appendix = "\n\n## 첨부 이미지\n" + "\n".join(f"[IMG:{role}]" for role in missing)
    return text.rstrip() + appendix


def _needs_interpretation_retry(text: str) -> bool:
    stripped = re.sub(r"\[IMG:[^\]]+\]", "", text)
    stripped = re.sub(r"#+\s*", "", stripped)
    stripped = stripped.strip()
    if len(stripped) < 120:
        return True

    section_hits = sum(
        1 for section in ("요약", "주요 영상 소견", "임상적 해석", "권고", "한계")
        if section in text
    )
    if section_hits < 3:
        return True

    sentence_count = len(re.findall(r"[.!?]\s+|[다요]\n|[다요]\s", stripped))
    return sentence_count < 4


def _build_interpretation_fallback(task: dict, step_results: list[dict], image_roles: list[str]) -> str:
    task_label = " / ".join(v for v in [task.get("department", ""), task.get("project", "")] if v) or "이번 검사"
    model_names = ", ".join(r.get("model", "") for r in step_results if r.get("model")) or "모델"
    result_types = ", ".join(sorted({r.get("result_type", "") for r in step_results if r.get("result_type")})) or "결과"
    token_preview = "\n".join(f"[IMG:{role}]" for role in _sort_image_roles(image_roles[:4]))
    if token_preview:
        token_preview = f"\n{token_preview}"

    return (
        f"## 전체 요약\n"
        f"{task_label}에 대해 {model_names}의 {result_types} 결과를 바탕으로 해석을 시도했습니다. "
        f"현재 자동 생성된 상세 분석이 충분하지 않아, 아래 요약은 모델 출력과 첨부 이미지를 기준으로 한 보수적 1차 해석입니다."
        f"{token_preview}\n\n"
        f"## 라벨별/구조별/클래스별 분석\n"
        f"첨부된 시각 자료는 모델이 분할 또는 탐지한 관심 영역의 위치와 범위를 보여줍니다. "
        f"구조의 정확한 형태, 경계, 분포는 원본 영상과 함께 확인해야 하며 단일 이미지나 일부 슬라이스만으로 단정적으로 판단해서는 안 됩니다.\n\n"
        f"## 임상적 의의\n"
        f"현재 결과는 관심 구조의 존재 여부와 상대적 분포를 파악하는 보조 자료로 해석하는 것이 적절합니다. "
        f"임상 증상, 원본 볼륨, 추가 판독 정보와 함께 종합할 때 의미가 커집니다.\n\n"
        f"## 권고사항\n"
        f"원본 영상 전체와 연속 슬라이스를 함께 검토하고, 필요한 경우 정량 지표 또는 전문의 판독으로 보완해 주세요. "
        f"자동 분할 결과만으로 진단을 확정하지 않는 것이 바람직합니다."
    )

def _parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    if "```" in raw:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"type": "knowledge", "message": raw}
