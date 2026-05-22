"""모델 초기 등록 스크립트 — ChromaDB가 비어있을 때 한 번 실행."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import httpx

BASE_URL = os.getenv("AGENT_URL", "http://localhost:8001")

MODELS = [
    {
        "id": "yolov12-si-joint-001",
        "model_name": "YOLOv12",
        "department": "Rheumatology",
        "project": "SI Joints Detection",
        "description": "Sacrum MRI에서 좌우 SI 관절 ROI 탐지. DICOM 입력을 받아 bounding box 좌표와 결과 이미지를 반환한다.",
        "task_type": "detection",
        "disease": "axSpA, 강직성 척추염",
        "required_data": ["dicom"],
        "result_type": "image",
    },
    {
        "id": "gradcam-bme-001",
        "model_name": "GradCAM++",
        "department": "Rheumatology",
        "project": "BME Classification",
        "description": "SI 관절 ROI에서 좌우 Bone Marrow Edema 분류. GradCAM 히트맵 이미지와 확률값을 반환한다.",
        "task_type": "classification",
        "disease": "axSpA, BME, 강직성 척추염",
        "required_data": ["dicom"],
        "result_type": "image",
    },
    {
        "id": "parkinson-gait-001",
        "model_name": "ParkinsonGait",
        "department": "Neurology",
        "project": "Parkinson Gait",
        "description": "보행 데이터(CSV)로 파킨슨 낙상 위험도를 분류한다. ExtraTrees 기반 ML 모델.",
        "task_type": "classification",
        "disease": "파킨슨병, 낙상 위험",
        "required_data": ["csv"],
        "result_type": "text",
    },
    {
        "id": "nnunet-smwi-001",
        "model_name": "nnUNet_SMWI",
        "department": "Neurology",
        "project": "nnUNet SMWI Segmentation",
        "description": "SMWI MRI에서 뇌 구조물 세그멘테이션. nnUNetv2 기반, 3D 렌더링 결과 이미지 반환.",
        "task_type": "segmentation",
        "disease": "신경퇴행성 질환, 뇌 구조물 이상",
        "required_data": ["dicom", "nifti"],
        "result_type": "image",
    },
]

def main():
    with httpx.Client(base_url=BASE_URL, timeout=30) as client:
        for m in MODELS:
            res = client.post("/agent/models/register", json=m)
            if res.status_code == 200:
                print(f"✓ {m['model_name']} 등록 완료")
            else:
                print(f"✗ {m['model_name']} 실패: {res.status_code} {res.text}")

if __name__ == "__main__":
    main()
