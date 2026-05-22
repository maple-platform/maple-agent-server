# maple-platform

의료 전문가가 AI 모델과 대화하듯 상호작용하는 **임상 AI 플랫폼**입니다.
도메인 특화 AI 모델 실행, RAG 기반 임상 지식 검색, VLM 범용 분석을 하나의 채팅 인터페이스에서 제공합니다.

---

## 시스템 구성

```
[maple-client]         React + Electron UI (Port 3000)
        │
        │  HTTP
        ▼
[maple-routing-server] FastAPI 백엔드 (Port 8000)
        │
        ├── HTTP ──────► [maple-model-execution-server] AI 모델 컨테이너 (Port 9001~9004)
        │
        └── SSH 터널 ──► [maple-agent-server] AI Agent (NHN Cloud B200, Port 8001)  ◄── 이 저장소
                                ├── vLLM  (Port 8003, gemma-4-31B-it, B200 ×2 텐서 병렬)
                                ├── ChromaDB  (Port 8002, 벡터 RAG)
                                └── /wiki  (마크다운 지식 누적 레이어)
```

| 저장소 | 역할 |
|---|---|
| `maple-client` | 채팅 UI, 추론 결과 시각화 (React + Electron) |
| `maple-routing-server` | 요청 라우팅, AI 컨테이너 오케스트레이션, 결과 저장 |
| **`maple-agent-server`** | **쿼리 분류, RAG 임상 해석, 모델 검색, VLM 분석** |
| `maple-model-execution-server` | 도메인 특화 AI 모델 컨테이너 (YOLOv12, GradCAM++, nnUNet 등) |

---

## 핵심 기능

### 1. 4-Mode 추론 라우팅
사용자 쿼리와 업로드 파일을 분석해 최적 모드로 자동 분기합니다.

| 모드 | 트리거 | 처리 |
|---|---|---|
| `prediction` | 의료 영상/데이터 + 특화 모델 쿼리 | ChromaDB 모델 검색 → 컨테이너 실행 → VLM 임상 해석 |
| `clinical` | 임상 지식 질문 | Wiki + RAG(PubMedQA·MedMCQA) → LLM 즉시 답변 |
| `general` | 이미지·수치 데이터 종합 분석 | VLM(gemma-4-31B-it) 멀티모달 분석 |
| `auto` | 기본값 | LLM이 쿼리 유형 판단 후 자동 분기 |

### 2. Wiki + RAG 하이브리드 지식 레이어

RAG 단독 방식과 달리, **매 해석 결과를 Wiki에 누적**하여 지식이 쌓일수록 해석 품질이 향상됩니다.

```
쿼리 수신
  ↓
Wiki index.md → 관련 모델/개념 페이지 파악
  ↓
ChromaDB 병렬 검색 (maple_models + maple_knowledge)
  ↓
gemma-4-31B-it → 임상 응답 생성
  ↓
해석 결과 → wiki/interpretations/ 에 파일링 (지식 누적)
```

| 레이어 | 역할 | 저장소 |
|---|---|---|
| Wiki | AI 모델 메타데이터, 임상 해석 패턴 누적 | `/wiki/*.md` |
| RAG | PubMedQA·MedMCQA 논문/QA 벡터 검색 | ChromaDB `maple_knowledge` |
| 모델 레지스트리 | 자연어 쿼리 기반 모델 탐색 | ChromaDB `maple_models` |

### 3. 다단계 AI 파이프라인

단일 요청으로 복수의 특화 모델이 순차 실행되는 파이프라인을 지원합니다.

```
예시: BME Classification
  Step 1 — YOLOv12      : MRI에서 SI 관절 ROI 탐지
  Step 2 — GradCAM++    : ROI 기반 BME 분류 + 히트맵 생성
  최종    — VLM 임상 해석 : 결과 이미지 + 예측값 → 임상 소견 텍스트 생성
```

### 4. VLM 임상 해석 with 이미지 인라인 렌더링

```
Agent 해석 출력:
  "좌측 SI 관절에 BME 소견이 확인됩니다 [IMG:gradcam_overlay] 우측은 정상 범위..."

프론트엔드:
  [IMG:gradcam_overlay] 토큰을 GradCAM 이미지로 인라인 치환하여 렌더링
```

---

## 기술 스택

| 분류 | 기술 | 세부 내용 |
|---|---|---|
| **LLM / VLM** | vLLM + gemma-4-31B-it | NHN Cloud B200 ×2, 텐서 병렬(TP=2), FP16 |
| **벡터 DB** | ChromaDB 0.5 | 모델 레지스트리 + 임상 논문 RAG |
| **임베딩** | sentence-transformers `all-MiniLM-L6-v2` | 쿼리·모델 설명 벡터화 |
| **AI 프레임워크** | FastAPI + asyncio | 전구간 비동기, 병렬 RAG 검색 |
| **특화 모델** | YOLOv12, GradCAM++, nnUNet, Parkinson Gait ML | Docker 컨테이너 격리 실행 |
| **프론트엔드** | React 19 + TypeScript + Electron | 웹/데스크탑 듀얼 빌드 |
| **백엔드** | FastAPI + Motor(MongoDB) + httpx | 비동기 DB, 비동기 컨테이너 호출 |
| **파일 처리** | pydicom, nibabel, Pillow | DICOM·NIfTI → PNG 변환, base64 인코딩 |

---

## 등록 AI 모델

| 모델 | 진료과 | 입력 | 출력 | 기술 |
|---|---|---|---|---|
| YOLOv12 | 류마티올로지 | DICOM | SI 관절 ROI 이미지 | Object Detection |
| GradCAM++ | 류마티올로지 | DICOM + ROI | BME 분류 + 히트맵 | Classification + XAI |
| Parkinson Gait ML | 신경과 | CSV (Gait 데이터) | 낙상 위험도 | Tabular ML |
| nnUNet SMWI | 신경과 | NIfTI | 뇌 구조 분할 | 3D Segmentation |

---

## API (maple-agent-server)

| Method | Endpoint | 설명 |
|---|---|---|
| `POST` | `/agent/plan` | 모드별 쿼리 라우팅 + 모델 실행 계획 수립 |
| `POST` | `/agent/interpret` | AI 추론 결과 VLM 임상 해석 |
| `POST` | `/agent/models/register` | 모델 등록 → Wiki + ChromaDB 자동 동기화 |
| `DELETE` | `/agent/models/{model_id}` | 모델 삭제 → Wiki + ChromaDB 자동 동기화 |
| `GET` | `/agent/models/lookup` | 자연어 쿼리로 모델 탐색 |
| `GET` | `/health` | vLLM + ChromaDB 연결 상태 |

---

## 빠른 시작

```bash
source maple-agent-venv/bin/activate

# 터미널 1 — ChromaDB
chroma run --host 0.0.0.0 --port 8002 --path ./agent-server/chroma_data

# 터미널 2 — vLLM (B200 ×2 텐서 병렬)
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-31B-it \
  --tensor-parallel-size 2 \
  --port 8003 \
  --max-model-len 8192

# 터미널 3 — Agent 서버
cd agent-server
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

> 최초 실행 시: `python agent-server/scripts/ingest_knowledge.py` (ChromaDB 임상 지식 초기화)

---

## 폴더 구조

```
maple-agent-server/
└── agent-server/
    ├── main.py                 # FastAPI 진입점 (Port 8001)
    ├── wiki/                   # 지식 누적 레이어
    │   ├── index.md            # 전체 Wiki 목록
    │   ├── models/             # AI 모델별 페이지
    │   ├── departments/        # 진료과별 페이지
    │   └── interpretations/    # 누적 임상 해석 패턴
    ├── services/
    │   ├── agent_service.py    # 핵심 오케스트레이션
    │   ├── wiki_service.py     # Wiki 읽기/쓰기
    │   └── rag_service.py      # ChromaDB 병렬 검색
    ├── llm/
    │   ├── client.py           # vLLM OpenAI 호환 클라이언트
    │   └── prompts.py          # 프롬프트 템플릿
    └── routers/
        ├── agent.py            # /agent/plan, /agent/interpret
        └── models.py           # /agent/models/*
```

---

## 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8003/v1` | vLLM API URL |
| `LLM_MODEL` | `google/gemma-4-31B-it` | 텍스트/VLM 모델명 |
| `CHROMA_HOST` | `localhost` | ChromaDB 호스트 |
| `CHROMA_PORT` | `8002` | ChromaDB 포트 |
| `WIKI_PATH` | `./wiki` | Wiki 파일 경로 |
