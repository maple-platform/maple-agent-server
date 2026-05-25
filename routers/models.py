import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import wiki_service
from rag import embedder

router = APIRouter()


class ModelRegisterRequest(BaseModel):
    id: str
    model_name: str
    department: str
    project: str
    description: str
    task_type: str
    disease: str
    required_data: list[str]
    result_type: str


@router.post("/register")
async def register_model(req: ModelRegisterRequest):
    # 1. Wiki 모델 페이지 생성/업데이트
    wiki_service.write_model_page(
        model_name=req.model_name,
        department=req.department,
        project=req.project,
        description=req.description,
        task_type=req.task_type,
        disease=req.disease,
        required_data=req.required_data,
        result_type=req.result_type,
    )

    # 2. 진료과 페이지 업데이트
    wiki_service.update_department_page(req.department, req.model_name, req.project)

    # 3. index.md 업데이트
    wiki_service.update_index(req.model_name, req.department, req.project)

    # 4. log.md 이력 추가
    wiki_service.append_log("ingest", f"{req.model_name} 모델 등록")

    # 5. ChromaDB maple_models 컬렉션에 등록
    doc_text = (
        f"모델명: {req.model_name}\n"
        f"진료과: {req.department}\n"
        f"프로젝트: {req.project}\n"
        f"설명: {req.description}\n"
        f"질환: {req.disease}\n"
        f"task_type: {req.task_type}\n"
        f"required_data: {', '.join(req.required_data)}\n"
        f"result_type: {req.result_type}"
    )
    embedder.upsert_model(
        model_id=req.id,
        text=doc_text,
        metadata={
            "model_name": req.model_name,
            "department": req.department,
            "project": req.project,
            "task_type": req.task_type,
            "required_data": ", ".join(req.required_data),
            "result_type": req.result_type,
        },
    )

    return {
        "status": "registered",
        "model_name": req.model_name,
        "wiki_page": f"wiki/models/{req.project}/{req.model_name}.md",
        "department_page": f"wiki/departments/{req.department}.md",
    }


@router.delete("/{model_id}")
async def delete_model(model_id: str):
    col = embedder._get_collection("maple_models")
    result = col.get(ids=[model_id], include=["metadatas"])
    if not result["ids"]:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found in registry")

    meta = result["metadatas"][0]
    model_name = meta.get("model_name", "")
    project = meta.get("project", "")

    wiki_service.delete_model_page(project, model_name)
    wiki_service.remove_model_from_index(project, model_name)
    wiki_service.append_log("delete", f"{project}/{model_name} 모델 삭제")

    embedder.delete_model(model_id)

    return {"status": "deleted", "model_id": model_id, "model_name": model_name, "project": project}


@router.get("/lookup")
async def lookup_model(model_name: str, project: str = ""):
    if not project:
        resolved = wiki_service.resolve_model(model_name)
        if not resolved:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found in wiki")
        project, model_name = resolved

    content = wiki_service.read_model_page(project, model_name)
    if not content:
        raise HTTPException(status_code=404, detail=f"Model {project}/{model_name} not found in wiki")

    department = ""
    for line in content.splitlines():
        if line.startswith("- **진료과:**"):
            department = re.sub(r'\*+', '', line.split(":", 1)[1]).strip()
            break

    return {"model_name": model_name, "project": project, "department": department}
