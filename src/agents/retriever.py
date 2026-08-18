"""
F3 — Retriever agent: берёт question из AgentState, ищет релевантные
куски в Qdrant, записывает их обратно в state["documents"].

Это узел графа LangGraph — на вход и выход всегда AgentState.
"""
import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import embeddings, qdrant_client
from ingestion import COLLECTION_NAME


def retriever_agent(state: dict, k: int = 4) -> dict:
    """
    Ищет top-k релевантных кусков документов по вопросу.
    Дописывает найденные куски в state["documents"] и лог в state["steps"].
    Если Qdrant недоступен (например, "уснул" бесплатный кластер) —
    не роняет весь граф, а честно сообщает об этом.
    """
    question = state["question"]

    try:
        query_vector = embeddings.embed_query(question)
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=k,
        )
        found_chunks = [point.payload["text"] for point in results.points]
        state["documents"] = found_chunks
        state.setdefault("steps", []).append(f"retriever: found {len(found_chunks)} chunks")

    except Exception as e:
        # Qdrant недоступен / кластер уснул / коллекция не найдена и т.п.
        state["documents"] = []
        state.setdefault("steps", []).append(f"retriever: error - {e}")

    return state


if __name__ == "__main__":
    from state import new_state

    test_state = new_state("Какая звезда самая массивная в этом документе?")
    result = retriever_agent(test_state)

    print("Вопрос:", result["question"])
    print("Найдено кусков:", len(result["documents"]))
    for i, doc in enumerate(result["documents"], 1):
        print(f"\n--- Кусок {i} ---")
        print(doc[:300])