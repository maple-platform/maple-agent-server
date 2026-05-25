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
        └── SSH 터널 ──► [maple-agent-server] AI Agent (NHN Cloud B200, Port 8101)  ◄── 이 저장소
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
예시: 뇌종양 분할
  Step 1 — BraTS2020_FLAIR_UNet3D : FLAIR MRI에서 뇌종양 영역 분할
  Step 2 — BraTS2020_T1ce_UNet3D  : T1ce MRI에서 Enhancing Tumor 분할
  최종    — VLM 임상 해석           : 분할 결과 이미지 → 임상 소견 텍스트 생성
```

### 4. VLM 임상 해석 with 이미지 인라인 렌더링

```
Agent 해석 출력:
  "우측 전두엽에 Whole Tumor 소견이 확인됩니다 [IMG:seg_overlay] Enhancing Tumor 영역은..."

프론트엔드:
  [IMG:seg_overlay] 토큰을 분할 결과 이미지로 인라인 치환하여 렌더링
```

---

## 기술 스택

| 분류 | 기술 | 세부 내용 |
|---|---|---|
| **LLM / VLM** | vLLM + gemma-4-31B-it | NHN Cloud B200 ×2, 텐서 병렬(TP=2), FP16 |
| **벡터 DB** | ChromaDB 0.5 | 모델 레지스트리 + 임상 논문 RAG |
| **임베딩** | sentence-transformers `all-MiniLM-L6-v2` | 쿼리·모델 설명 벡터화 |
| **AI 프레임워크** | FastAPI + asyncio | 전구간 비동기, 병렬 RAG 검색 |
| **특화 모델** | UNet3D(BraTS), YOLO26x(RSNA), ChestXray14 | Docker 컨테이너 격리 실행 |
| **프론트엔드** | React 19 + TypeScript + Electron | 웹/데스크탑 듀얼 빌드 |
| **백엔드** | FastAPI + Motor(MongoDB) + httpx | 비동기 DB, 비동기 컨테이너 호출 |
| **파일 처리** | pydicom, nibabel, Pillow | DICOM·NIfTI → PNG 변환, base64 인코딩 |

---

## 등록 AI 모델

| 모델 | 진료과 | 입력 | 출력 | 기술 |
|---|---|---|---|---|
| BraTS2020_FLAIR_UNet3D | 신경과 | NIfTI (FLAIR) | 뇌종양 분할 오버레이 | 3D Segmentation |
| BraTS2020_T1_UNet3D | 신경과 | NIfTI (T1) | 뇌종양 분할 오버레이 | 3D Segmentation |
| BraTS2020_T1ce_UNet3D | 신경과 | NIfTI (T1ce) | 뇌종양 분할 오버레이 | 3D Segmentation |
| BraTS2020_T2_UNet3D | 신경과 | NIfTI (T2) | 뇌종양 분할 오버레이 | 3D Segmentation |
| ChestXray14_Multilabel_Classification | 영상의학과 | PNG/JPG | GradCAM 오버레이 + 14개 흉부 소견 확률 | Classification + XAI |
| YOLO26x_RSNA_Pneumonia | 영상의학과 | DICOM | 폐렴 의심 영역 bbox 오버레이 | Object Detection |

---

## 폴더 구조

```
maple-agent-server/
├── main.py                 # FastAPI 진입점 (Port 8101)
├── requirements.txt
├── .env.example
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
├── rag/
│   ├── embedder.py         # ChromaDB upsert / delete / get
│   └── retriever.py        # ChromaDB query + 결과 포맷
└── routers/
    ├── agent.py            # /agent/plan, /agent/interpret
    └── models.py           # /agent/models/*
```

---

## 빠른 시작

```bash
source maple-agent-venv/bin/activate

# 터미널 1 — ChromaDB
chroma run --host 0.0.0.0 --port 8002 --path ./chroma_data

# 터미널 2 — vLLM (B200 ×2 텐서 병렬)
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-31B-it \
  --tensor-parallel-size 2 \
  --port 8003 \
  --max-model-len 8192

# 터미널 3 — Agent 서버
uvicorn main:app --host 0.0.0.0 --port 8101 --reload
```

### 최초 실행 시 (1회)

```bash
# 1. 임상 지식 베이스 구축 — PubMedQA·MedMCQA → ChromaDB maple_knowledge
python scripts/ingest_knowledge.py

# 2. AI 모델 등록 — AI_Models/ 스캔 → MongoDB + ChromaDB maple_models
#    (maple-model-execution-server 디렉토리에서 실행)
cd ../maple-model-execution-server
python scan_and_register.py
```

---

## API

### Agent

| Method | Endpoint | 설명 |
|---|---|---|
| `POST` | `/agent/plan` | 모드별 쿼리 라우팅 + 모델 실행 계획 수립 |
| `POST` | `/agent/interpret` | AI 추론 결과 VLM 임상 해석 |

### Models

| Method | Endpoint | 설명 |
|---|---|---|
| `POST` | `/agent/models/register` | 모델 등록 → Wiki + ChromaDB 자동 동기화 |
| `DELETE` | `/agent/models/{model_id}` | 모델 삭제 → Wiki + ChromaDB 자동 동기화 |
| `GET` | `/agent/models/lookup` | 자연어 쿼리로 모델 탐색 |
| `GET` | `/health` | vLLM + ChromaDB 연결 상태 |

### POST `/agent/plan`

**Request**

```json
{
  "query": "천장관절 Detection 해줘",
  "mode": "auto",
  "uploaded_types": ["dicom"],
  "history": [],
  "images": [],
  "csv_data": []
}
```

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `query` | string | 필수 | 사용자 자연어 요청 |
| `mode` | string | `"auto"` | `auto` \| `clinical` \| `prediction` \| `general` |
| `uploaded_types` | string[] | `[]` | 업로드된 파일 타입 목록 (e.g. `["dicom"]`) |
| `images` | string[] | `[]` | `general` 모드 — DICOM/NIfTI를 PNG로 변환한 base64 목록 |
| `csv_data` | dict[] | `[]` | `general` 모드 — `pd.read_csv().to_dict("records")` 결과 |

**모드별 동작**

| mode | Agent 처리 |
|---|---|
| `prediction` | ChromaDB `maple_models` 검색 → required_data 매칭 → 실행 계획 반환 |
| `clinical` | Wiki + RAG(PubMedQA·MedMCQA) 병렬 검색 → LLM 즉시 답변 |
| `general` | 이미지(VLM) + CSV 수치 데이터 → gemma-4-31B-it 종합 분석 |
| `auto` | images/csv_data 있으면 `general`; 없으면 LLM이 `prediction` / `clinical` / `general` 판단 |

**Response**

```json
// prediction — 실행 계획 수립
{
  "status": "ready",
  "mode": "prediction",
  "execution_plan": {
    "steps": [{"step": 1, "model": "YOLOv12", "department": "Rheumatology", "project": "SI Joints Detection"}]
  }
}

// prediction — 파일 미첨부
{"status": "requires_input", "mode": "prediction", "message": "..."}

// prediction — 모델 미매칭
{"status": "no_model", "mode": "prediction", "message": "..."}

// prediction — 파일 타입 불일치
{"status": "type_mismatch", "mismatched_models": [{"model": "YOLOv12", "required": ["dicom"], "uploaded": ["csv"]}]}

// clinical
{"query_type": "knowledge", "mode": "clinical", "message": "...", "sources": [...], "model_suggestion": "..."}

// general
{"query_type": "general", "mode": "general", "message": "..."}
```

### POST `/agent/interpret`

**Request**

```json
{
  "query": "BME Classification 해줘",
  "task": {"department": "Rheumatology", "project": "BME Classification"},
  "execution_context": {"mode": "prediction", "plan": {}},
  "step_results": [
    {
      "step": 1,
      "model": "GradCAM++",
      "result_type": "image",
      "predictions": {"left": {"prob": 0.94, "pred": 1}, "right": {"prob": 0.65, "pred": 1}},
      "images": [{"role": "gradcam_overlay", "data": "data:image/png;base64,..."}]
    }
  ]
}
```

**Response**

```json
{
  "interpretation": "## 전체 요약\n좌측 SI관절에 BME가 확인되었으며...\n\n[IMG:gradcam_overlay]",
  "interpretation_raw": "## 전체 요약\n좌측 SI관절에 BME가 확인되었으며...\n\n[IMG:gradcam_overlay]",
  "images": {"gradcam_overlay": "data:image/png;base64,..."}
}
```

> `[IMG:role]` 토큰 위치에 `images` 맵의 이미지를 인라인 렌더링합니다.

**자동 재시도 및 폴백**

| 조건 | 처리 |
|---|---|
| VLM HTTP 오류 | 이미지 없이 텍스트 전용 LLM 재시도 |
| 품질 부족 (텍스트 < 120자, 섹션 < 3개, 문장 < 4개) | 동일 이미지로 1회 재생성 |
| 재시도 후에도 품질 부족 | 서버 측 폴백 텍스트 반환 |

### POST `/agent/models/register`

```json
{
  "id": "radiology-rsna-pneumonia-yolo26x",
  "model_name": "YOLO26x_RSNA_Pneumonia",
  "department": "Radiology",
  "project": "RSNA_Pneumonia_YOLO26x",
  "description": "폐렴성 폐 혼탁 탐지 모델. DICOM 흉부 X-ray를 입력받아 폐렴 의심 영역을 bounding box로 검출",
  "task_type": "bbox detection",
  "required_data": ["dicom"],
  "result_type": "image"
}
```

→ `wiki/models/RSNA_Pneumonia_YOLO26x/YOLO26x_RSNA_Pneumonia.md` 생성 + `wiki/index.md` 업데이트 + ChromaDB `maple_models` 등록 자동 수행

---

## 추론 흐름

### prediction 모드

```
POST /agent/plan {mode: "prediction", query, uploaded_types}
  ↓ ChromaDB maple_models 검색 (유사도 0.40 미만 제거) → required_data 매칭
  ← {status: "ready", execution_plan: {steps: [...]}}

(백엔드가 AI 컨테이너 실행 후)

POST /agent/interpret {query, task, step_results}
  ↓ Wiki 페이지 수집 → 이미지 role 매핑
  ↓ VLM(180s) 또는 LLM(120s) → 품질 검증 → 필요 시 재시도
  ↓ 해석 결과 wiki/interpretations/ 에 누적
  ← {interpretation, images}
```

### clinical 모드

```
POST /agent/plan {mode: "clinical", query}
  ↓ Wiki 키워드 검색 + ChromaDB maple_models·maple_knowledge 병렬 검색
  ↓ LLM 임상 답변 생성
  ← {message, sources, model_suggestion}
```

### general 모드

```
POST /agent/plan {mode: "general", query, images, csv_data}
  ↓ 이미지 있으면 VLM, 없으면 LLM → 종합 분석
  ← {message}
```

---

## Wiki 스키마

### index.md

```markdown
# Maple AI Agent Wiki Index

## Models
- [[YOLOv12]] - Rheumatology/SI Joints Detection

## Departments
- [[Rheumatology]] - SI Joints Detection

## Interpretations
- [[BME_pattern_001]] - 좌측 BME 고확률 패턴
```

### 모델 페이지 (`wiki/models/ModelName.md`)

```markdown
# ModelName

## 기본 정보
- **진료과:** Rheumatology
- **task_type:** detection
- **required_data:** [dicom]

## 임상 해석 패턴
### [2026-04-15] 해석 패턴
누적된 해석 결과...
```

---

## 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8003/v1` | vLLM OpenAI 호환 API URL |
| `LLM_MODEL` | `google/gemma-4-31B-it` | 텍스트/VLM 모델명 |
| `CHROMA_HOST` | `localhost` | ChromaDB 호스트 |
| `CHROMA_PORT` | `8002` | ChromaDB 포트 |
| `WIKI_PATH` | `./wiki` | Wiki 파일 경로 |

---

## 트러블슈팅

### ChromaDB tenant 오류

```
ValueError: Tenant default_tenant not found
```

ChromaDB를 먼저 실행한 뒤 `ingest_knowledge.py`로 초기화합니다.

```bash
chroma run --host 0.0.0.0 --port 8002 --path ./chroma_data
python scripts/ingest_knowledge.py
```

### vLLM 시작 시 GPU 메모리 부족

```
ValueError: Free memory on device cuda:1 is less than desired GPU memory utilization
```

다른 프로세스가 GPU를 점유 중입니다. `nvidia-smi`로 확인 후 해당 프로세스를 종료하세요.

### vLLM gemma4 아키텍처 인식 불가

```
model type `gemma4` but Transformers does not recognize this architecture
```

```bash
pip install --upgrade transformers
```

### `no_model` 반환 — 모델이 등록되어 있는데 검색 실패

1. ChromaDB 실행 상태 확인
2. `POST /agent/models/register` 정상 호출 여부 확인
3. 쿼리가 모델 설명과 너무 달라 유사도 0.40 미만인 경우 쿼리를 더 구체적으로 작성
