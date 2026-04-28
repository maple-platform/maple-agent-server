# MARS AI Agent - 구현 지시서

## 프로젝트 개요
MARS AI 플랫폼의 AI Agent 서버.
워크스테이션 백엔드(FastAPI, Port 8000)로부터 추론 결과를 받아 임상 해석하고, 모델 실행 계획을 수립한다.

## 시스템 구성
```
[워크스테이션]                          [NHN Cloud B200]
MARS_AI_Back-end (Port 8000)  ──SSH터널──►  MARS_AI_Agent (Port 8001)  ◄── 이 서버
                                            ├── Ollama (Port 11434)
                                            │   └── gemma4:31b
                                            ├── ChromaDB (벡터 DB, RAG용)
                                            └── /wiki (LLM Wiki, 지식 누적용)
```

## 접속 정보
- **작업 경로:** `/NHNHOME/WORKSPACE/0226010256_A/mars-platform/mars-ai-agent`
- **SSH 터널:** `localhost:8001` → NHN Cloud `8001`, `localhost:11435` → Ollama `11434`

---

## 핵심 아키텍처: Wiki + RAG 하이브리드

### 설계 철학
기존 RAG 단독 방식은 매 쿼리마다 지식을 재발견한다. 이 서버는 **LLM Wiki 패턴**(Karpathy, 2026)과 **RAG**를 결합하여 지식이 누적·컴파일되는 구조를 채택한다.

### 역할 분담
| 레이어 | 담당 | 저장소 |
|---|---|---|
| **Wiki** | AI 모델 메타데이터, 핵심 의학 개념, 임상 해석 패턴 누적 | `/wiki/*.md` |
| **RAG** | PubMedQA·MedMCQA 대용량 논문/QA 검색 | ChromaDB (`mars_knowledge`) |
| **모델 레지스트리** | 등록된 AI 모델 검색 | ChromaDB (`mars_models`) |

### 쿼리 처리 흐름
```
사용자 질의
    ↓
1. Wiki index.md 확인 → 관련 모델/개념 페이지 파악
    ↓
2. (필요시) RAG → ChromaDB에서 논문/QA 검색 보완
    ↓
3. Gemma4:31b → Wiki + RAG 결과 합쳐서 응답 생성
    ↓
4. 좋은 응답/분석 → Wiki에 파일링 (지식 누적)
```

---

## 폴더 구조
```
mars-ai-agent/
├── main.py
├── requirements.txt
├── .env
├── CLAUDE.md
│
├── wiki/                      # LLM Wiki (지식 누적 레이어)
│   ├── index.md               # 전체 wiki 목록 및 요약 (항상 최신 유지)
│   ├── log.md                 # 작업 이력 (append-only)
│   ├── models/                # AI 모델별 페이지
│   ├── departments/           # 진료과별 페이지
│   ├── concepts/              # 의학 개념 페이지
│   └── interpretations/       # 누적된 임상 해석 패턴
│
├── services/
│   ├── agent_service.py       # 핵심 오케스트레이션
│   ├── wiki_service.py        # Wiki 읽기/쓰기/업데이트
│   └── rag_service.py         # ChromaDB RAG 검색
│
├── llm/
│   ├── client.py              # Ollama 비동기 클라이언트
│   └── prompts.py             # 프롬프트 템플릿
│
├── rag/
│   ├── embedder.py            # ChromaDB upsert
│   └── retriever.py           # ChromaDB query
│
└── routers/
    ├── agent.py               # /agent/* 엔드포인트
    └── models.py              # /agent/models/* 엔드포인트
```

---

## Wiki 스키마 규칙

### index.md 형식
```markdown
# MARS AI Agent Wiki Index

## Models
- [[YOLOv12]] - Rheumatology/SI Joints Detection, DICOM detection
- [[GradCAM++]] - Rheumatology/BME Classification, DICOM classification

## Departments
- [[Rheumatology]] - SI Joints Detection, BME Classification
- [[Neurology]] - Parkinson Gait, nnUNet SMWI

## Concepts
- [[axSpA]] - 축성 척추관절염
- [[BME]] - Bone Marrow Edema

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
이전 단계, 다음 단계 정보

## 임상 해석 패턴
누적된 해석 패턴들

## 관련 개념
[[axSpA]], [[BME]]
```

### log.md 형식
```markdown
## [2026-04-15] ingest | YOLOv12 모델 등록
## [2026-04-15] interpret | BME Classification 결과 해석
## [2026-04-15] lint | wiki 일관성 검사
```

---

## API 엔드포인트

### POST `/agent/plan`
모델 실행 계획 수립

**Request:**
```json
{
  "query": "천장관절 Detection 해줘",
  "uploaded_types": ["dicom"],
  "history": []
}
```

**Response (모델 실행):**
```json
{
  "status": "ready",
  "execution_plan": {
    "steps": [
      {"step": 1, "model": "YOLOv12", "department": "Rheumatology", "project": "SI Joints Detection"}
    ]
  },
  "missing_inputs": [],
  "message": "실행 계획이 수립되었습니다."
}
```

**Response (의학 지식 질문):**
```json
{
  "query_type": "knowledge",
  "message": "axSpA는 축성 척추관절염으로...",
  "sources": [
    {"source": "wiki", "page": "axSpA", "score": 0.95},
    {"source": "pubmedqa", "question": "...", "score": 0.87}
  ]
}
```

### POST `/agent/interpret`
추론 결과 임상 해석 (백엔드에서 호출)

**Request:**
```json
{
  "query": "BME Classification 해줘",
  "step_results": [
    {
      "step": 1,
      "model": "GradCAM++",
      "result_type": "image",
      "predictions": {"left": {"prob": 0.94, "pred": 1}, "right": {"prob": 0.65, "pred": 1}}
    }
  ]
}
```

**Response:**
```json
{
  "interpretation": "좌측 SI관절에 BME가 확인되었으며 확률은 94%입니다..."
}
```

### POST `/agent/models/register`
모델 등록 → Wiki 페이지 자동 생성 + ChromaDB 등록

**Request:**
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

### DELETE `/agent/models/{model_id}`
모델 삭제 → Wiki 페이지 업데이트 + ChromaDB 삭제

### GET `/agent/models/lookup?model_name=YOLOv12`
모델명으로 진료과/프로젝트 조회

### GET `/health`
헬스체크

---

## 환경변수 (.env)
```
OLLAMA_URL=http://localhost:11434
LLM_MODEL=gemma4:31b
CHROMA_HOST=localhost
CHROMA_PORT=8002
MARS_BACKEND_URL=http://localhost:8000
WIKI_PATH=./wiki
```

## 기술 스택
- **웹 프레임워크:** FastAPI + Uvicorn
- **LLM:** Ollama (gemma4:31b)
- **벡터 DB:** ChromaDB (RAG용, mars_models + mars_knowledge)
- **Wiki:** 마크다운 파일 (`/wiki`)
- **임베딩:** sentence-transformers
- **HTTP 클라이언트:** httpx (비동기)

## 구현 우선순위
1. FastAPI 기본 구조 + health check
2. Ollama 클라이언트 (gemma4:31b)
3. Wiki 서비스 (읽기/쓰기/index 업데이트)
4. 모델 등록 API (`/agent/models/register`) → Wiki 자동 생성
5. ChromaDB RAG 서비스
6. `/agent/plan` - 쿼리 분류 + Wiki 검색 + RAG 검색
7. `/agent/interpret` - 추론 결과 해석 + Wiki 파일링