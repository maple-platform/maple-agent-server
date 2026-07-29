"""Clinical Board 실행 함수 모음 (v4 2.3).

각 _run_* 는 프롬프트 조립 → LLM/VLM 호출 → JSON 파싱을 담당한다.
호출부(agent_service.interpret 오케스트레이터)에서 try/except로 감싸므로,
여기서는 예외를 잡지 않고 그대로 전파한다(fail-safe는 오케스트레이터 책임).
"""

import json
import logging

from services import wiki_service
from services.rag_service import retrieve
from llm import client as llm_client
from llm.prompts import (
    SYSTEM_READER,
    SYSTEM_CHALLENGER,
    SYSTEM_EVIDENCE,
    SYSTEM_GUARDIAN,
    build_reader_prompt,
    build_challenger_prompt,
    build_evidence_prompt,
    build_guardian_prompt,
)

logger = logging.getLogger("maple-ai-agent")


def _parse_json(raw: str) -> dict:
    """LLM 응답에서 JSON 오브젝트를 추출. 실패는 오케스트레이터까지 전파한다."""
    raw = (raw or "").strip()
    if "```" in raw:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("[board] JSON parse failed; raw head=%s", raw[:200])
        raise ValueError("board response is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("board response must be a JSON object")
    return parsed


def _require_list(parsed: dict, key: str) -> list:
    value = parsed.get(key)
    if not isinstance(value, list):
        raise ValueError(f"board response field '{key}' must be a list")
    return value


async def _run_reader(
    query: str,
    task: dict,
    execution_context: dict,
    step_results: list[dict],
    wiki_context: str,
    raw_images: list[str],
    image_roles: list[str],
) -> dict:
    """Reader — 모델 output 기반 3필드 초안 + claims/sentence_map."""
    prompt = build_reader_prompt(
        query, task, execution_context, step_results, wiki_context,
        image_roles=image_roles,
    )
    if raw_images:
        raw = await llm_client.generate_with_images(prompt, raw_images, system=SYSTEM_READER)
    else:
        raw = await llm_client.generate(prompt, system=SYSTEM_READER)

    parsed = _parse_json(raw)
    result = {
        "finding": parsed.get("finding", "") or "",
        "interpretation": parsed.get("interpretation", "") or "",
        "recommendation": parsed.get("recommendation", "") or "",
        "claims": _require_list(parsed, "claims"),
        "sentence_map": _require_list(parsed, "sentence_map"),
    }
    if not any(result[key].strip() for key in ("finding", "interpretation", "recommendation")):
        raise ValueError("reader returned no human-facing content")
    return result


async def _run_challenger(
    query: str,
    task: dict,
    step_results: list[dict],
    wiki_context: str,
    raw_images: list[str],
    alt_model_output: dict | None = None,
) -> dict:
    """Challenger — 대체 모델 output 기반(있으면) 또는 블라인드 동일모델 + RAG DDx 체크리스트.

    Reader의 opinion/claims는 전달하지 않는다(블라인드 원칙)."""
    ddx_context = ""
    if alt_model_output is None:
        # 블라인드 폴백: 감별진단 체크리스트를 RAG로 조회
        try:
            _models, _knowledge, ddx_context = await retrieve(query)
        except Exception:
            logger.exception("[board] challenger DDx retrieve failed")
            ddx_context = ""

    prompt = build_challenger_prompt(
        query, task, step_results, wiki_context,
        alt_model_output=alt_model_output,
        ddx_context=ddx_context,
    )
    if raw_images:
        raw = await llm_client.generate_with_images(prompt, raw_images, system=SYSTEM_CHALLENGER)
    else:
        raw = await llm_client.generate(prompt, system=SYSTEM_CHALLENGER)

    parsed = _parse_json(raw)
    return {
        "differential": _require_list(parsed, "differential"),
        "source": "alt_model" if alt_model_output is not None else "blind_same_model",
    }


async def _run_evidence(
    query: str,
    reader_claims: list[dict],
    challenger_differential: list[dict],
) -> dict:
    """Evidence — Reader.claims ∪ Challenger.differential 주장 단위 RAG 대조."""
    try:
        _models, _knowledge, rag_context = await retrieve(query)
    except Exception:
        logger.exception("[board] evidence retrieve failed")
        rag_context = ""

    prompt = build_evidence_prompt(query, reader_claims, challenger_differential, rag_context)
    raw = await llm_client.generate(prompt, system=SYSTEM_EVIDENCE)

    parsed = _parse_json(raw)
    return {
        "evidence_map": _require_list(parsed, "evidence_map"),
        "unsupported_claims": _require_list(parsed, "unsupported_claims"),
    }


async def _run_guardian(
    query: str,
    task: dict,
    step_results: list[dict],
    reader_out: dict,
    challenger_out: dict,
    evidence_out: dict,
    raw_images: list[str],
) -> dict:
    """Guardian — 자체 RAG 재조회 + 최종 veto."""
    try:
        _models, _knowledge, guardian_rag_context = await retrieve(query)
    except Exception:
        logger.exception("[board] guardian retrieve failed")
        guardian_rag_context = ""

    prompt = build_guardian_prompt(
        query, task, step_results, reader_out, challenger_out, evidence_out, guardian_rag_context,
    )
    if raw_images:
        raw = await llm_client.generate_with_images(prompt, raw_images, system=SYSTEM_GUARDIAN)
    else:
        raw = await llm_client.generate(prompt, system=SYSTEM_GUARDIAN)

    parsed = _parse_json(raw)
    risk_tier = parsed.get("risk_tier")
    if risk_tier not in {"low", "moderate", "high", "critical"}:
        raise ValueError("guardian returned invalid risk_tier")
    flags = _require_list(parsed, "flags")
    evidence_refs = _require_list(parsed, "evidence_refs")
    rationale = parsed.get("rationale", "")
    if not isinstance(rationale, str):
        raise ValueError("guardian rationale must be a string")
    if risk_tier in {"high", "critical"} and not rationale.strip():
        raise ValueError("high/critical risk requires a rationale")
    return {
        "veto": bool(parsed.get("veto", False)),
        "risk_tier": risk_tier,
        "flags": flags,
        "rationale": rationale,
        "evidence_refs": evidence_refs,
    }
