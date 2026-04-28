from typing import Any, Literal

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field, field_validator

from services import agent_service

router = APIRouter()


class PlanRequest(BaseModel):
    query: str
    uploaded_types: list[str] = Field(default_factory=list)   # ["dicom", "csv", ...] 업로드된 파일 타입 목록
    history: list[dict] = Field(default_factory=list)
    mode: Literal["auto", "clinical", "prediction", "general"] = "auto"
    # general 모드용 데이터 (백엔드가 변환해서 넘김)
    images: list[str] = Field(default_factory=list)           # DICOM/NIfTI → PNG 변환 후 base64
    csv_data: list[dict] = Field(default_factory=list)        # CSV → pd.read_csv().to_dict() 결과

    @field_validator("uploaded_types", "history", "images", "csv_data", mode="before")
    @classmethod
    def normalize_optional_lists(cls, value):
        return [] if value is None else value


class ImageResult(BaseModel):
    role: str        # bbox_overlay | gradcam_overlay | segmentation_overlay
    data: str        # data:image/png;base64,...


class StepResult(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    step: int
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


class InterpretRequest(BaseModel):
    query: str
    task: TaskInfo = TaskInfo(department="", project="")
    execution_context: ExecutionContext = ExecutionContext(mode="prediction")
    step_results: list[StepResult]


@router.post("/plan")
async def plan(req: PlanRequest):
    # prediction 모드에서 파일 없이 텍스트만 오는 경우 차단
    if req.mode == "prediction" and not req.uploaded_types:
        return {
            "status": "requires_input",
            "mode": "prediction",
            "message": (
                "AI 예측 모드는 분석할 의료 데이터(DICOM 등)가 필요합니다. "
                "파일을 첨부하거나, 의학 지식 질문은 임상 모드(clinical)로 변경해 주세요."
            ),
        }

    return await agent_service.plan(
        query=req.query,
        uploaded_types=req.uploaded_types,
        history=req.history,
        mode=req.mode,
        images=req.images,
        csv_data=req.csv_data,
    )


@router.post("/interpret")
async def interpret(req: InterpretRequest):
    return await agent_service.interpret(
        query=req.query,
        task=req.task.model_dump(),
        execution_context=req.execution_context.model_dump(),
        step_results=[s.model_dump() for s in req.step_results],
    )
