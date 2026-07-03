import time
from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field, field_validator

from services import agent_service
from services.experiment_logger import append_experiment_record

router = APIRouter()


class Attachment(BaseModel):
    """라우팅 서버가 정규화한 첨부파일 하나.
    (input-normalization-design.md §3 계약: attachments[])"""
    type: str = ""                                            # dicom | nifti | image | csv | pdf ...
    filename: str = ""
    images: list[str] = Field(default_factory=list)           # VLM 입력 이미지 (data URI). 없으면 []
    text: str = ""                                            # VLM 입력 텍스트 (ASR/OCR/추출). 없으면 ""
    metadata: dict = Field(default_factory=dict)              # 구조화·비식별화 메타데이터
    tabular: list[dict] | None = None                        # CSV면 dict 리스트, 아니면 null

    @field_validator("images", mode="before")
    @classmethod
    def _images_none_to_list(cls, value):
        return [] if value is None else value

    @field_validator("metadata", mode="before")
    @classmethod
    def _metadata_none_to_dict(cls, value):
        return {} if value is None else value


class PlanRequest(BaseModel):
    query: str
    uploaded_types: list[str] = Field(default_factory=list)   # ["dicom", "csv", ...] 업로드된 파일 타입 목록
    history: list[dict] = Field(default_factory=list)
    mode: Literal["auto", "clinical", "prediction", "general"] = "auto"
    # general 모드용 데이터 (백엔드가 변환해서 넘김)
    images: list[str] = Field(default_factory=list)           # (레거시) DICOM/NIfTI → PNG 변환 후 base64
    csv_data: list[dict] = Field(default_factory=list)        # (레거시) CSV → pd.read_csv().to_dict() 결과
    attachments: list[Attachment] = Field(default_factory=list)  # 정규화 첨부파일 (레거시 images/csv_data 대체)

    @field_validator("uploaded_types", "history", "images", "csv_data", "attachments", mode="before")
    @classmethod
    def normalize_optional_lists(cls, value):
        return [] if value is None else value


class ImageResult(BaseModel):
    role: str        # bbox_overlay | gradcam_overlay | segmentation_overlay
    data: str        # data:image/png;base64,...


class StepResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    step: int | str        # prediction=순번(int), general DAG=step_id(str "s1")
    model: str
    result_type: str = ""
    predictions: Any = Field(default_factory=list)
    model_output: Any = Field(default_factory=dict)      # ROI 좌표, 분류 상세, segmentation 메타, raw text 등
    images: list[ImageResult | dict | str] = Field(default_factory=list)

    @field_validator("images", mode="before")
    @classmethod
    def normalize_images(cls, value):
        return [] if value is None else value


class TaskInfo(BaseModel):
    department: str
    project: str


class ExecutionContext(BaseModel):
    mode: str
    plan: dict = {}              # /agent/plan이 반환한 execution_plan
    attachments_meta: list[dict] = Field(default_factory=list)   # 원본 메타 (전환기)
    attachments: list[dict] = Field(default_factory=list)        # 원본 스캔 (이미지+메타)


class InterpretRequest(BaseModel):
    query: str
    task: TaskInfo = TaskInfo(department="", project="")
    execution_context: ExecutionContext = ExecutionContext(mode="prediction")
    step_results: list[StepResult]


@router.post("/plan")
async def plan(req: PlanRequest):
    # RSNA QI EXPERIMENT LOGGING START: timing-only side effect; safe to remove after study.
    started = time.perf_counter()
    # RSNA QI EXPERIMENT LOGGING END

    # prediction 모드에서 파일 없이 텍스트만 오는 경우 차단
    if req.mode == "prediction" and not req.uploaded_types and not req.attachments:
        result = {
            "status": "requires_input",
            "mode": "prediction",
            "message": (
                "AI 예측 모드는 분석할 의료 데이터(DICOM 등)가 필요합니다. "
                "파일을 첨부하거나, 의학 지식 질문은 임상 모드(clinical)로 변경해 주세요."
            ),
        }
    else:
        result = await agent_service.plan(
            query=req.query,
            uploaded_types=req.uploaded_types,
            history=req.history,
            mode=req.mode,
            images=req.images,
            csv_data=req.csv_data,
            attachments=[a.model_dump() for a in req.attachments],
        )

    # RSNA QI EXPERIMENT LOGGING START: remove this block after study if desired.
    steps = result.get("execution_plan", {}).get("steps", []) if isinstance(result, dict) else []
    append_experiment_record({
        "endpoint": "plan",
        "query": req.query,
        "mode": req.mode,
        "uploaded_types": req.uploaded_types,
        "latency_sec": round(time.perf_counter() - started, 4),
        "status": result.get("status") if isinstance(result, dict) else None,
        "response_mode": result.get("mode") if isinstance(result, dict) else None,
        "selected_model": steps[0].get("model") if steps else None,
        "step_count": len(steps),
    })
    # RSNA QI EXPERIMENT LOGGING END

    return result


@router.post("/interpret")
async def interpret(req: InterpretRequest):
    # RSNA QI EXPERIMENT LOGGING START: timing-only side effect; safe to remove after study.
    started = time.perf_counter()
    result = await agent_service.interpret(
        query=req.query,
        task=req.task.model_dump(),
        execution_context=req.execution_context.model_dump(),
        step_results=[s.model_dump() for s in req.step_results],
    )
    interpretation = result.get("interpretation", "") if isinstance(result, dict) else ""
    append_experiment_record({
        "endpoint": "interpret",
        "query": req.query,
        "latency_sec": round(time.perf_counter() - started, 4),
        "task": req.task.model_dump(),
        "step_models": [s.model for s in req.step_results],
        "interpretation_generated": bool(interpretation),
        "xai_marker_included": "[IMG:" in interpretation,
        "image_count": len(result.get("images", {})) if isinstance(result, dict) else 0,
    })
    # RSNA QI EXPERIMENT LOGGING END

    return result
