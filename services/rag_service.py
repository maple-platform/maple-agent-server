import asyncio
from rag.retriever import search_models, search_knowledge, build_rag_context


async def retrieve(query: str) -> tuple[list[dict], list[dict], str]:
    """모델 레지스트리 + 지식 베이스 검색을 병렬로 실행 후 컨텍스트 반환"""
    def _search_models():
        try:
            return search_models(query, n_results=3)
        except Exception:
            return []

    def _search_knowledge():
        try:
            return search_knowledge(query, n_results=5)
        except Exception:
            return []

    loop = asyncio.get_event_loop()
    model_results, knowledge_results = await asyncio.gather(
        loop.run_in_executor(None, _search_models),
        loop.run_in_executor(None, _search_knowledge),
    )

    context = build_rag_context(model_results, knowledge_results)
    return model_results, knowledge_results, context
