# H100 단독 서버 배포 노트 (agent-server only)

> 이 문서는 **agent-server만** 올리는 H100 단독 서버(`ai-s-c16-33`) 배포 기록이다.
> README의 실행 절차는 원본 **B200 ×2 (TP=2)** 기준이며, 이 서버는 **H100 ×1 (TP=1)** 이라 차이가 있어 별도 정리한다.
> routing-server·model-execution-server는 별도 A100 서버 담당. 여기는 chroma + vLLM + FastAPI app만 상주.
> 최초 세팅: 2026-07-13.

## 토폴로지 (이 서버 부분)

```
maple-client ─▶ routing-server(:8100, A100) ─┬─▶ agent-server (이 H100, :8101)
                                              │       ├── vLLM (:8011, gemma-4-31B-it, TP=1, bf16)
                                              │       └── ChromaDB (:8010, 벡터 RAG)
                                              └─▶ model-execution (:9020~, A100)
```

## 경로 배치

| 용도 | 경로 |
|---|---|
| 코드 (dev 브랜치) | `/home/tta/maple-agent-server` |
| conda / HF캐시 / chroma / 로그 | `/data/maple/` (1.5TB 볼륨) |
| conda 설치 | `/data/maple/miniconda` |
| ChromaDB 영속 | `/data/maple/chroma_data` |
| HF 모델 캐시 | `/data/maple/hf_cache` |
| 로그 | `/data/maple/logs/{vllm,chroma,agent,ingest}.log` |
| 기동 스크립트 | `/data/maple/start_{vllm,chroma,app}.sh` |

## conda 환경 — **2개로 분리 (중요)**

한 env에 몰면 vLLM이 pydantic을 2.13으로 올려 **chromadb 0.5.20 서버가 깨진다** (등록 시 500, create_collection 파싱 실패).
app은 vLLM을 import하지 않고 HTTP로만 호출하므로 분리해도 무방하다.

| env | 용도 | 핵심 핀 |
|---|---|---|
| `maple-agent` | vLLM 서버 전용 | vLLM 0.25.0, torch 2.11+cu130, pydantic 2.13 |
| `maple-app` | FastAPI app + ChromaDB | chromadb 0.5.20, **pydantic 2.9.2**, sentence-transformers 3.2.1 |

> 근본 원인: `requirements.txt`의 `vllm`이 unpinned → 최신 0.25.0이 `pydantic>=2.12` 요구.
> 후속 권장: `vllm==0.25.0` 핀 or requirements에서 vLLM 분리.

## GPU 드라이버

- 기존 535(CUDA 12.2) → 최신 torch/vLLM이 GPU 인식 못 함 → **580(CUDA 13.0)으로 apt 업그레이드**.
- 리부팅 없이 `modprobe -r`/`modprobe`로 모듈 리로드 적용 (GPU 점유 프로세스 없을 때).
  - 점유하던 `dcgm-exporter` docker 컨테이너를 잠깐 내렸다가 580 기준으로 재생성함.

## 기동 (tmux 상주)

```bash
# 순서: chroma → vllm → app  (ingest는 최초 1회)
tmux new -d -s chroma "bash /data/maple/start_chroma.sh 2>&1 | tee /data/maple/logs/chroma.log"
tmux new -d -s vllm   "bash /data/maple/start_vllm.sh   2>&1 | tee /data/maple/logs/vllm.log"
tmux new -d -s agent  "bash /data/maple/start_app.sh    2>&1 | tee /data/maple/logs/agent.log"
```

`start_vllm.sh` 핵심 (H100 1장 대응):
- `--tensor-parallel-size 1`, `--max-model-len 8192`, bf16 (기본)  — GPU 78.9/81.5GB (빡빡)
- `CUDA_HOME=$ENV/lib/python3.10/site-packages/nvidia/cu13` + PATH에 ninja(env/bin)  — flashinfer JIT용 nvcc/ninja
- `VLLM_USE_FLASHINFER_SAMPLER=0`  — 툴킷 헤더 버전 불일치로 flashinfer 샘플러 JIT가 깨져서 native 샘플링 사용
- ffmpeg 시스템 설치 필요 (torchcodec의 libavutil)
- 모델 `google/gemma-4-31B-it`은 public/apache-2.0 → HF 토큰 불필요

## 최초 1회 세팅

```bash
conda activate maple-app
# 모델 등록 (옵션 A: agent 단독, routing/MongoDB 불필요) — 6개 등록
AGENT_URL=http://localhost:8101 python scripts/register_models.py
# 임상 지식 RAG (clinical 모드용, 오래 걸림)
python scripts/ingest_knowledge.py
```

## 검증 (2026-07-13 통과)

```bash
curl localhost:8101/health          # {"status":"ok","vllm":"ok","chromadb":"ok"}
```

- LLM 스모크: vLLM `/v1/chat/completions` → 정확한 응답 ✅
- `prediction` 모드: nii.gz + 뇌종양 쿼리 → BraTS DAG 계획 ✅
- `general` 모드(에이전틱): 흉부 X-ray 폐렴 쿼리 → 의도분석 → YOLO+ChestXray14 모델 선택 → DAG 계획(`depends_on`) ✅
  - 매칭 모델 없으면 `fallback_vlm_only` 경로로 VLM 단독 분석 ✅
- `interpret`(3단계): 합성 step_results(gradcam+원본 스캔) → **VLM + 모델 Wiki 결합** 임상 해석 생성 ✅
- `clinical` 모드: `ingest_knowledge` 완료 후 RAG 답변 가능

### 모드별 VLM / RAG / Wiki 사용 (코드 기준)

| 경로 | VLM(이미지) | Wiki | RAG(pubmedqa) | 비고 |
|---|:---:|:---:|:---:|---|
| general 폴백 (`_run_general_vlm`) | ✅ | ✕ | ✕ | 모델 미매칭 시 이미지만 보고 VLM 분석 |
| **interpret (3단계)** | ✅ | ✅(모델 Wiki 페이지) | ✕ | 모델 결과+원본 스캔+Wiki를 VLM에 결합 → 임상 해석 |
| auto (`_auto`) | general로 위임 | ✅ | ✅ | wiki+RAG를 계획 프롬프트에 주입 |
| clinical (`_clinical`) | ✕(텍스트) | ✅ | ✅ | wiki+RAG → 텍스트 임상 답변 |

> 즉 "VLM이 볼 때 Wiki도 함께"는 **interpret(3단계)**에서 일어난다. RAG(pubmedqa)는 VLM 이미지 경로엔 안 들어가고 auto/clinical의 텍스트 답변에만 쓰인다.

## 알려진 함정 요약

1. 드라이버 535로는 GPU 인식 불가 → 580 필수.
2. app+vLLM 한 env → pydantic 충돌로 chromadb 등록 500. env 2개 분리로 해결.
3. flashinfer 샘플러 JIT는 CUDA 툴킷 헤더 불일치로 실패 → `VLLM_USE_FLASHINFER_SAMPLER=0`.
4. torchcodec 로드 실패 → 시스템 ffmpeg 설치.

## 네트워크 / 서버 간 통신

| 호스트 | 역할 | 공인 IP | 사설 IP (`ens224`) |
|---|---|---|---|
| `ai-s-c16-23-01` | 라우팅 + 모델실행 (A100 ×1) | 123.41.22.219 | 10.70.16.2 |
| `ai-s-c16-33` (이 서버) | agent + vLLM + ChromaDB (H100 ×1) | 123.41.23.124 | 10.70.16.3 |

- 두 GPU 서버는 **사설망(`10.70.16.x`, `ens224`)으로 인터넷을 경유하지 않고 직접 통신**한다.
- 라우팅 서버 → agent 호출은 사설 IP로 설정: `AGENT_URL=http://10.70.16.3:8101` (기존 SSH 터널 대체).
- 사설 IP 확인: `ifconfig | grep ens224 -A1`
- 사설망 SSH: `ssh tta@10.70.16.3`
- 클라이언트는 로컬(노트북)에서 실행되어 라우팅 서버(서버 A)로 요청을 보낸다.
