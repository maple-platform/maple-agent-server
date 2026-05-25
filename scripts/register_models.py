"""모델 초기 등록 스크립트 — ChromaDB가 비어있을 때 한 번 실행."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import httpx

BASE_URL = os.getenv("AGENT_URL", "http://localhost:8101")

MODELS = []

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
