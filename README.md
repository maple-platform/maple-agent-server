# MARS AI Agent

MARS AI 플랫폼의 AI Agent 서버입니다.
백엔드(MARS_AI_Back-end, Port 8000)로부터 추론 요청을 받아 모델 실행 계획을 수립하고,
AI 모델 추론 결과를 임상적으로 해석하며, 의학 지식 질문에 답변합니다.

---

## 시스템 구성

MARS AI 플랫폼은 세 개의 독립적인 서버로 구성됩니다.

```
[Frontend UI]  mars-agent-v2 (React, Port 3000)
      │
      │  HTTP
      ▼
[Back-end]     MARS_AI_Back-end (FastAPI, Port 8000)
      │
      │  HTTP (SSH터널 → localhost:8001)
      ▼
[AI Agent]     MARS_AI_Agent (NHN Cloud B200 x2, Port 8001)    ◄── 이 저장소
               ├── vLLM (Port 8003, gemma4:31B-it, tensor-parallel-size 2)
               ├── ChromaDB (Port 8002, RAG)
               └── /wiki (LLM Wiki, 지식 누적)
```

| 서버 | 저장소 | 포트 | 역할 |
|---|---|---|---|
| Back-end | `MARS_AI_Back-end` | 8000 | 추론 라우팅, 프로젝트 관리, 결과 저장, 파일 변환 |
| **AI Agent** | **`MARS_AI_Agent`** | **NHN Cloud B200:8001 (SSH터널 → localhost:8001)** | **모드별 쿼리 라우팅, RAG 임상 해석, 모델 검색, VLM 범용 분석** |
| Frontend | `mars-agent-v2` | 3000 | 사용자 인터페이스 |

---

## 기술 스택

| 분류 | 기술 | 버전 | 용도 |
|---|---|---|---|
| **웹 프레임워크** | FastAPI | 0.115.0 | Agent API 서버 |
| | Uvicorn | 0.30.6 | ASGI 서버 |
| | Pydantic | 2.9.2 | 요청/응답 스키마 검증 |
| **LLM / VLM** | vLLM + gemma-4-31B-it | 0.20.0+ | 임상 해석 생성, 범용 멀티모달 분석 (B200 x2 텐서 병렬) |
| **벡터 DB** | ChromaDB | 0.5.20 | 모델 레지스트리 + 논문/QA RAG 검색 |
| **임베딩** | sentence-transformers | 3.2.1 | 모델 설명·논문 벡터화 (`all-MiniLM-L6-v2`) |
| **HTTP 클라이언트** | httpx | 0.27.2 | vLLM OpenAI 호환 API 비동기 호출 |
| **Wiki** | 마크다운 파일 (`/wiki`) | - | 모델 메타데이터·임상 해석 패턴 누적 |

---

## 핵심 아키텍처: Wiki + RAG 하이브리드

### 설계 철학

기존 RAG 단독 방식은 매 쿼리마다 지식을 재발견합니다. 이 서버는 **LLM Wiki 패턴**과 **RAG**를 결합하여 지식이 누적·컴파일되는 구조를 채택합니다.

### 역할 분담

| 레이어 | 담당 | 저장소 |
|---|---|---|
| **Wiki** | AI 모델 메타데이터, 핵심 의학 개념, 임상 해석 패턴 누적 | `/wiki/*.md` |
| **RAG** | PubMedQA·MedMCQA 대용량 논문/QA 검색 | ChromaDB (`mars_knowledge`) |
| **모델 레지스트리** | 등록된 AI 모델 검색 | ChromaDB (`mars_models`) |

### 쿼리 처리 흐름

```
백엔드 요청
    ↓
1. Wiki index.md 확인 → 관련 모델/개념 페이지 파악
    ↓
2. RAG → ChromaDB mars_models + mars_knowledge 병렬 검색
    ↓
3. gemma-4-31B-it → Wiki + RAG 결과 합쳐서 응답 생성
    ↓
4. 좋은 응답/분석 → Wiki에 파일링 (지식 누적)
```

---

## 폴더 구조

```
mars-platform-private/          # 모노레포 루트
├── .gitignore
├── README.md
│
└── agent-server/               # AI Agent 서버 (이 패키지)
    ├── main.py                 # FastAPI 앱 진입점 (Port 8001)
    ├── requirements.txt
    ├── .env.example
    │
    ├── wiki/                   # LLM Wiki (지식 누적 레이어)
    │   ├── index.md            # 전체 wiki 목록 및 요약 (항상 최신 유지)
    │   ├── log.md              # 작업 이력 (append-only)
    │   ├── models/             # AI 모델별 페이지
    │   ├── departments/        # 진료과별 페이지
    │   ├── concepts/           # 의학 개념 페이지
    │   └── interpretations/    # 누적된 임상 해석 패턴
    │
    ├── services/
    │   ├── agent_service.py    # 핵심 오케스트레이션 (plan / interpret)
    │   ├── wiki_service.py     # Wiki 읽기/쓰기/업데이트
    │   └── rag_service.py      # ChromaDB RAG 병렬 검색 (asyncio.gather)
    │
    ├── llm/
    │   ├── client.py           # vLLM OpenAI 호환 비동기 클라이언트
    │   └── prompts.py          # 프롬프트 템플릿
    │
    ├── rag/
    │   ├── embedder.py         # ChromaDB upsert / delete / get
    │   └── retriever.py        # ChromaDB query + 결과 포맷
    │
    └── routers/
        ├── agent.py            # /agent/plan, /agent/interpret
        └── models.py           # /agent/models/*
```

---

## API 엔드포인트

### Agent — 추론 계획 및 임상 해석

| Method | Endpoint | 설명 |
|---|---|---|
| `POST` | `/agent/plan` | 모드별 쿼리 라우팅 및 모델 실행 계획 수립 |
| `POST` | `/agent/interpret` | AI 모델 추론 결과 임상 해석 |

### Models — 모델 레지스트리

| Method | Endpoint | 설명 |
|---|---|---|
| `POST` | `/agent/models/register` | 모델 등록 → Wiki 페이지 생성 + ChromaDB 등록 |
| `DELETE` | `/agent/models/{model_id}` | 모델 삭제 → Wiki 페이지 삭제 + ChromaDB 제거 |
| `GET` | `/agent/models/lookup?model_name=YOLOv12` | 모델명으로 진료과/프로젝트 조회 |

### Health

| Method | Endpoint | 설명 |
|---|---|---|
| `GET` | `/health` | 헬스체크 (vLLM + ChromaDB 연결 상태 포함) |

---

## 스키마 상세

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
| `uploaded_types` | string[] | `[]` | 업로드된 파일 확장자 목록 (e.g. `["dicom", "csv"]`) |
| `history` | dict[] | `[]` | 대화 이력 (현재 미사용) |
| `images` | string[] | `[]` | `general` 모드용 — DICOM/NIfTI를 PNG로 변환한 base64 문자열 목록 |
| `csv_data` | dict[] | `[]` | `general` 모드용 — `pd.read_csv().to_dict("records")` 결과 |

**모드별 동작**

| mode | Agent 처리 |
|---|---|
| `prediction` | ChromaDB `mars_models` 검색 → required_data 매칭 → 실행 계획 반환 |
| `clinical` | Wiki + RAG(PubMedQA·MedMCQA) 병렬 검색 → LLM 즉시 답변 |
| `general` | 이미지(VLM) + CSV 수치 데이터 → gemma-4-31B-it 종합 분석 |
| `auto` | `images` / `csv_data` 있으면 `general`로 분기; 없으면 LLM이 쿼리 분석 후 `execution` / `knowledge` / `general` 중 판단 |

**Response — prediction 모드**

```json
// 실행 계획 수립 완료
{
  "status": "ready",
  "mode": "prediction",
  "execution_plan": {
    "steps": [
      {
        "step": 1,
        "model": "YOLOv12",
        "department": "Rheumatology",
        "project": "SI Joints Detection",
        "task_type": "detection",
        "result_type": "image"
      }
    ]
  },
  "missing_inputs": [],
  "message": "1개 모델 실행 계획이 수립되었습니다."
}
```

```json
// 파일 미첨부
{"status": "requires_input", "mode": "prediction", "message": "AI 예측 모드는 분석할 의료 데이터(DICOM 등)가 필요합니다..."}

// 유사도 점수 0.40 미만 또는 검색 결과 없음
{"status": "no_model", "mode": "prediction", "message": "요청에 맞는 등록된 AI 모델을 찾지 못했습니다..."}

// required_data와 uploaded_types 불일치
{
  "status": "type_mismatch",
  "mode": "prediction",
  "message": "...",
  "mismatched_models": [
    {"model": "YOLOv12", "required": ["dicom"], "uploaded": ["csv"]}
  ]
}
```

| status | 발생 조건 |
|---|---|
| `ready` | 매칭된 모델이 있고 데이터 타입도 일치 |
| `requires_input` | `prediction` 모드인데 `uploaded_types`가 비어 있음 |
| `no_model` | ChromaDB 검색 결과 없음 또는 유사도 점수 0.40 미만 |
| `type_mismatch` | 모델이 요구하는 `required_data`와 업로드 파일 타입 불일치 |

**Response — clinical 모드**

```json
{
  "query_type": "knowledge",
  "mode": "clinical",
  "message": "axSpA는 축성 척추관절염으로...",
  "sources": [
    {"source": "pubmedqa", "text": "...", "score": 0.91}
  ],
  "model_suggestion": "관련 질환에 사용할 수 있는 특화 AI 모델이 있습니다: SI Joints Detection (YOLOv12)"
}
```

> `model_suggestion`은 RAG 검색에서 관련 특화 모델이 발견된 경우에만 포함됩니다 (없으면 `null`).

**Response — general 모드**

```json
{
  "query_type": "general",
  "mode": "general",
  "message": "이미지와 수치 데이터를 종합 분석한 결과..."
}
```

**Response — auto 모드**

LLM 판단 결과에 따라 `prediction` / `clinical` / `general` 응답 형식 중 하나를 반환합니다.

---

### POST `/agent/interpret`

**Request**

```json
{
  "query": "BME Classification 해줘",
  "task": {
    "department": "Rheumatology",
    "project": "BME Classification"
  },
  "execution_context": {
    "mode": "prediction",
    "plan": {}
  },
  "step_results": [
    {
      "step": 1,
      "model": "GradCAM++",
      "result_type": "image",
      "predictions": {"left": {"prob": 0.94, "pred": 1}, "right": {"prob": 0.65, "pred": 1}},
      "model_output": {},
      "images": [
        {"role": "gradcam_overlay", "data": "data:image/png;base64,..."}
      ]
    }
  ]
}
```

| 파라미터 | 타입 | 설명 |
|---|---|---|
| `query` | string | 원래 사용자 요청 |
| `task.department` | string | 진료과명 |
| `task.project` | string | 프로젝트명 |
| `execution_context.mode` | string | 실행 모드 |
| `execution_context.plan` | dict | `/agent/plan`이 반환한 execution_plan |
| `step_results[].step` | int | 단계 번호 |
| `step_results[].model` | string | 모델명 |
| `step_results[].result_type` | string | `image` \| `text` 등 |
| `step_results[].predictions` | any | 모델 예측 결과 (확률, 클래스 등) |
| `step_results[].model_output` | any | ROI 좌표, segmentation 메타, raw text 등 |
| `step_results[].images[].role` | string | 이미지 역할 식별자 (e.g. `bbox_overlay`, `gradcam_overlay`, `segmentation_3d`) |
| `step_results[].images[].data` | string | `data:image/png;base64,...` 형식 |

**Response**

```json
{
  "interpretation": "## 전체 요약\n좌측 SI관절에 BME가 확인되었으며...\n\n[IMG:gradcam_overlay]",
  "interpretation_raw": "## 전체 요약\n좌측 SI관절에 BME가 확인되었으며...\n\n[IMG:gradcam_overlay]",
  "images": {
    "gradcam_overlay": "data:image/png;base64,..."
  }
}
```

> `interpretation`과 `interpretation_raw`는 동일한 값입니다. 둘 다 `[IMG:role]` 마커가 포함된 raw 마크다운 텍스트입니다.
> `images`는 `role → "data:image/png;base64,..."` 매핑입니다.
> `[IMG:role]`을 `<figure><img>` 태그로 치환하는 HTML 변환은 백엔드에서 `images` 맵을 이용해 수행합니다.

**자동 재시도 및 폴백 로직**

| 조건 | 처리 |
|---|---|
| VLM HTTP 오류 (`HTTPStatusError`) | 이미지 없이 텍스트 전용 LLM 재시도 |
| VLM 네트워크 오류 (`HTTPError`) | 텍스트 전용 LLM 재시도 |
| 응답 품질 부족 (텍스트 < 120자, 섹션 < 3개, 문장 < 4개) | 동일 이미지로 1회 재생성 |
| 재시도 후에도 품질 부족 | 서버 측 폴백 텍스트 생성 |

---

### POST `/agent/models/register`

**Request**

```json
{
  "id": "yolov12-si-joint-001",
  "model_name": "YOLOv12",
  "department": "Rheumatology",
  "project": "SI Joints Detection",
  "description": "Sacrum MRI에서 좌우 SI 관절 ROI 탐지",
  "task_type": "detection",
  "disease": "axSpA, 강직성 척추염",
  "required_data": ["dicom"],
  "result_type": "image"
}
```

→ 자동으로:
1. `wiki/models/YOLOv12.md` 생성/업데이트
2. `wiki/departments/Rheumatology.md` 업데이트
3. `wiki/index.md` 업데이트
4. `wiki/log.md`에 이력 추가
5. ChromaDB `mars_models` 컬렉션에 등록

**Response**

```json
{
  "status": "registered",
  "model_name": "YOLOv12",
  "wiki_page": "wiki/models/YOLOv12.md",
  "department_page": "wiki/departments/Rheumatology.md"
}
```

---

### DELETE `/agent/models/{model_id}`

**Response**

```json
{
  "status": "deleted",
  "model_id": "yolov12-si-joint-001",
  "model_name": "YOLOv12"
}
```

→ 자동으로:
1. `wiki/models/YOLOv12.md` 삭제
2. `wiki/index.md`에서 해당 항목 제거
3. `wiki/log.md`에 이력 추가
4. ChromaDB `mars_models`에서 제거

---

### GET `/agent/models/lookup`

**Query parameter:** `model_name=YOLOv12`

**Response**

```json
{
  "model_name": "YOLOv12",
  "department": "Rheumatology",
  "project": "SI Joints Detection"
}
```

---

### GET `/health`

**Response**

```json
{
  "status": "ok",
  "service": "mars-ai-agent",
  "vllm": "ok",
  "chromadb": "ok",
  "model": "google/gemma-4-31B-it"
}
```

| 필드 | 값 |
|---|---|
| `status` | `ok` (vLLM + ChromaDB 모두 정상) \| `degraded` (하나 이상 불가) |
| `vllm` | `ok` \| `unavailable` |
| `chromadb` | `ok` \| `unavailable` |
| `model` | 현재 설정된 LLM 모델명 |

---

## 파일 타입 정규화

백엔드가 전송하는 파일 확장자를 Agent가 표준 포맷명으로 정규화합니다.

| 백엔드 전송값 | 정규화 결과 |
|---|---|
| `"dcm"`, `"dicom"`, `"ima"` | `"dicom"` |
| `"nii"`, `"nii.gz"` | `"nifti"` |
| `"hdr"`, `"img"` | `"analyze"` |
| `"mgh"`, `"mgz"` | `"mgh"` |
| `"mnc"` | `"minc"` |
| `"nrrd"`, `"nhdr"`, `"seg.nrrd"` | `"nrrd"` |
| `"mha"`, `"mhd"` | `"metaimage"` |
| `"svs"`, `"ndpi"`, `"scn"`, `"mrxs"`, `"ome.tif"` | `"wsi"` |
| `"png"`, `"jpg"`, `"jpeg"`, `"bmp"` | `"image"` |
| `"tif"`, `"tiff"` | `"tiff"` |
| `"csv"` | `"csv"` |
| `"xlsx"`, `"xls"` | `"excel"` |
| `"npy"`, `"npz"` | `"numpy"` |
| `"h5"`, `"hdf5"` | `"hdf5"` |
| `"gz"` (단독) | 정규화 없음 (모델이 `nifti` 요구 시 조건부 허용) |

> NIfTI 조건부 처리: 모델의 `required_data`에 `"nifti"`가 포함된 경우, `"gz"` 단독 확장자도 자동 허용합니다.

---

## 추론 흐름 (Agent 관점)

### prediction 모드

```
POST /agent/plan {mode: "prediction", query, uploaded_types}
    ↓
Agent: ChromaDB mars_models에서 쿼리와 관련된 모델 검색 (최대 5개)
       유사도 점수 0.40 미만 제거
       모델의 required_data와 uploaded_types 매칭
    ↓
← {status: "ready", execution_plan: {steps: [...]}} 반환

(백엔드가 AI 모델 컨테이너를 실행하고 결과를 수집한 뒤)

POST /agent/interpret {query, task, execution_context, step_results}
    ↓
Agent: 관련 모델 Wiki 페이지 수집 (최대 600자)
       step_results에서 이미지 추출 → role 매핑 + raw base64 리스트 구성
       이미지가 있으면 VLM(generate_with_images, timeout 180s)
       없으면 LLM(generate, timeout 120s)으로 해석 생성
       응답 품질 검증 → 필요 시 자동 재시도 또는 폴백
       해석 결과를 wiki/models/{model_name}.md에 누적
    ↓
← {interpretation, interpretation_raw, images} 반환
```

### clinical 모드

```
POST /agent/plan {mode: "clinical", query}
    ↓
Agent: wiki_service.search_wiki(query) — 키워드 매칭, 최대 3개 페이지 반환
       rag_service.retrieve(query) — ChromaDB mars_models(3개) + mars_knowledge(5개) 병렬 검색
       LLM으로 임상 답변 생성
    ↓
← {query_type: "knowledge", mode: "clinical", message, sources, model_suggestion} 반환
```

### general 모드

```
POST /agent/plan {mode: "general", query, images, csv_data}
    images: 백엔드가 DICOM/NIfTI를 PNG로 변환한 base64 문자열 목록
    csv_data: 백엔드가 pd.read_csv().to_dict("records")로 파싱한 결과
    ↓
Agent: 이미지가 있으면 VLM(generate_with_images), 없으면 LLM(generate)으로 종합 분석
       VLM 실패 시 텍스트 전용 LLM 폴백
    ↓
← {query_type: "general", mode: "general", message} 반환
```

### auto 모드

```
POST /agent/plan {mode: "auto", query, uploaded_types}
    ↓
Agent: request에 images / csv_data가 있으면 → general 분기
       없으면: Wiki + RAG 병렬 검색 후 LLM이 쿼리 유형 판단
         → "execution" : prediction 응답 반환
         → "knowledge" : clinical 응답 형식으로 반환
         → "general"   : _general(query, [], [])으로 재분기
```

---

## Wiki 스키마 규칙

### index.md 형식

```markdown
# MARS AI Agent Wiki Index

## Models
- [[YOLOv12]] - Rheumatology/SI Joints Detection
- [[GradCAM++]] - Rheumatology/BME Classification

## Departments
- [[Rheumatology]] - SI Joints Detection
- [[Neurology]] - Parkinson Gait, nnUNet SMWI

## Concepts
- [[axSpA]] - 축성 척추관절염

## Interpretations
- [[BME_pattern_001]] - 좌측 BME 고확률 패턴
```

### 모델 페이지 형식 (`wiki/models/ModelName.md`)

```markdown
# ModelName

## 기본 정보
- **진료과:** Rheumatology
- **프로젝트:** SI Joints Detection
- **task_type:** detection
- **required_data:** [dicom]
- **result_type:** image

## 설명
모델에 대한 설명

## 파이프라인
(등록 후 업데이트)

## 임상 해석 패턴
### [2026-04-15] 해석 패턴
누적된 해석 결과...

## 관련 개념
axSpA, 강직성 척추염
```

### log.md 형식

```markdown
## [2026-04-15] ingest | YOLOv12 모델 등록
## [2026-04-15] interpret | YOLOv12 결과 해석 완료
## [2026-04-15] delete | GradCAM++ 모델 삭제
```

---

## 환경 변수

`.env` 파일 또는 환경변수로 설정합니다. `.env.example` 참고.

| 변수 | 기본값 | 설명 |
|---|---|---|
| `LLM_BASE_URL` | `http://localhost:8003/v1` | vLLM OpenAI 호환 API URL |
| `LLM_MODEL` | `google/gemma-4-31B-it` | 텍스트 생성 모델명 (HuggingFace ID) |
| `VLM_MODEL` | `google/gemma-4-31B-it` | 이미지 포함 VLM 모델명 |
| `LLM_MAX_TOKENS` | `2048` | 최대 생성 토큰 수 |
| `CHROMA_HOST` | `localhost` | ChromaDB 호스트 |
| `CHROMA_PORT` | `8002` | ChromaDB 포트 |
| `WIKI_PATH` | `./wiki` | Wiki 마크다운 파일 경로 |

---

## 실행 방법

**사전 요구사항**
- Python 3.10+
- vLLM 0.20.0+, transformers 5.5.0+
- ChromaDB (`localhost:8002`)
- NVIDIA GPU (B200 권장, FP16/BF16 지원)

```bash
cd agent-server

# 의존성 설치
pip install -r requirements.txt

# 1. vLLM 서버 시작 (별도 터미널, B200 x2 텐서 병렬)
python -m vllm.entrypoints.openai.api_server \
  --model google/gemma-4-31B-it \
  --tensor-parallel-size 2 \
  --port 8003 \
  --max-model-len 8192

# 2. ChromaDB 서버 시작 (별도 터미널)
chroma run --host 0.0.0.0 --port 8002 --path ./chroma_data

# 3. Agent 서버 실행
python main.py
```

API 문서: http://localhost:8001/docs

**ChromaDB 초기 데이터 ingest (최초 1회)**

```bash
python scripts/ingest_knowledge.py
```

---

## 트러블슈팅

### vLLM 시작 시 GPU 메모리 부족

```
ValueError: Free memory on device cuda:1 is less than desired GPU memory utilization
```

다른 프로세스(Ollama 등)가 GPU 메모리를 점유하고 있습니다.

```bash
# 점유 프로세스 확인
nvidia-smi
# 해당 프로세스 종료 후 vLLM 재시작
```

### vLLM 시작 시 gemma4 아키텍처 인식 불가

```
model type `gemma4` but Transformers does not recognize this architecture
```

transformers 버전이 낮습니다.

```bash
pip install --upgrade transformers
```

---

### ChromaDB tenant 오류

```
ValueError: Tenant default_tenant not found
```

Agent 서버 최초 실행 시 ChromaDB를 먼저 띄운 후 `ingest_knowledge.py`를 실행하여 초기화합니다.

```bash
chroma run --host 0.0.0.0 --port 8002 --path ./chroma_data
python scripts/ingest_knowledge.py
```

---

### prediction 모드에서 `type_mismatch` 반환

```json
{"status": "type_mismatch", "mismatched_models": [...]}
```

업로드한 파일 타입이 모델의 `required_data`와 맞지 않습니다.
`GET /agent/models/lookup?model_name={모델명}`으로 모델 정보를 조회하거나, 올바른 파일 형식을 사용하세요.

---

### NIfTI(`.nii.gz`) 파일인데 `type_mismatch` 반환

백엔드가 `.nii.gz` 파일에서 마지막 확장자 `"gz"`만 전송하는 경우, Agent는 다음 규칙으로 처리합니다.

- 모델의 `required_data`에 `"nifti"`가 포함되어 있으면 `"gz"` 확장자도 자동 허용합니다.
- 모델의 `required_data`에 `"nifti"`가 없으면 `type_mismatch`가 발생합니다.

해결: 백엔드에서 복합 확장자(`.nii.gz`)를 파싱해 `uploaded_types=["nifti"]`로 전송하거나, 모델 등록 시 `required_data`에 `"nifti"`를 포함시킵니다.

---

### `no_model` 반환 — 모델이 등록되어 있는데도 검색 실패

1. ChromaDB가 실행 중인지 확인합니다.
2. 모델 등록 시 `POST /agent/models/register`가 정상 호출되었는지 확인합니다.
3. `GET /agent/models/lookup?model_name=YOLOv12`로 Wiki에 모델 페이지가 있는지 확인합니다.
4. 쿼리 문장이 모델 설명과 너무 달라 유사도 점수 0.40 미만인 경우, 쿼리를 더 구체적으로 작성합니다.

---

### VLM 해석 품질이 낮거나 이미지 토큰이 누락됨

Agent는 다음 조건을 만족하지 못하면 자동으로 재생성을 시도합니다.

- 해석 텍스트 (이미지 토큰 제외) 120자 미만
- 필수 섹션 (요약, 주요 영상 소견, 임상적 해석, 권고) 중 3개 미만 포함
- 문장 수 4개 미만

재시도 후에도 품질이 부족하면 서버 측 폴백 텍스트를 반환합니다.
vLLM 타임아웃 (`generate`: 120초, `generate_with_images`: 180초)이 너무 짧은 경우,
`llm/client.py`의 `timeout` 값을 늘리는 것을 고려하세요.
