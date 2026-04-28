from rag.retriever import search_models, search_knowledge, build_rag_context


async def retrieve(query: str) -> tuple[list[dict], list[dict], str]:
    """모델 레지스트리 + 지식 베이스 검색 후 컨텍스트 반환"""
    try:
        model_results = search_models(query, n_results=3)
    except Exception:
        model_results = []

    try:
        knowledge_results = search_knowledge(query, n_results=5)
    except Exception:
        knowledge_results = []

    context = build_rag_context(model_results, knowledge_results)
    return model_results, knowledge_results, context
