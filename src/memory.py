"""
F10 — Long-term memory: хранит пары (вопрос, ответ) из прошлых диалогов
в отдельной коллекции Qdrant. Перед ответом на новый вопрос система
вспоминает релевантные прошлые обмены репликами и добавляет их в
state["history"], чтобы ответы были последовательными между сессиями.
"""
import os
import sys
import uuid
import time

sys.path.append(os.path.dirname(__file__))

from config import embeddings, qdrant_client
from qdrant_client.models import VectorParams, Distance, PointStruct

MEMORY_COLLECTION = "memory"


def ensure_memory_collection():
    """Создаёт коллекцию памяти, если её ещё нет (не пересоздаёт существующую)."""
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if MEMORY_COLLECTION in existing:
        return

    # Узнаём размерность вектора текущей модели эмбеддингов
    sample_vector = embeddings.embed_query("test")
    vector_size = len(sample_vector)

    qdrant_client.create_collection(
        collection_name=MEMORY_COLLECTION,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def save_memory(question: str, answer: str):
    """Сохраняет пару (вопрос, ответ) в память после успешного ответа."""
    ensure_memory_collection()

    text = f"Вопрос: {question}\nОтвет: {answer}"
    vector = embeddings.embed_query(text)

    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=vector,
        payload={
            "question": question,
            "answer": answer,
            "timestamp": time.time(),
        },
    )
    qdrant_client.upsert(collection_name=MEMORY_COLLECTION, points=[point])


def recall_memory(question: str, k: int = 3) -> list:
    """Находит релевантные прошлые обмены репликами по новому вопросу."""
    ensure_memory_collection()

    query_vector = embeddings.embed_query(question)
    results = qdrant_client.query_points(
        collection_name=MEMORY_COLLECTION,
        query=query_vector,
        limit=k,
    )

    memories = []
    for point in results.points:
        payload = point.payload
        memories.append(f"Ранее спрашивали: {payload['question']}\nОтветили: {payload['answer']}")

    return memories


def recall_node(state: dict) -> dict:
    """Узел графа: подтягивает релевантную память в начале работы."""
    question = state["question"]
    memories = recall_memory(question)

    state["history"] = memories
    state.setdefault("steps", []).append(f"memory: recalled {len(memories)} past exchanges")

    return state


def save_node(state: dict) -> dict:
    """Узел графа: сохраняет итоговый вопрос-ответ в память в конце."""
    question = state["question"]
    answer = state.get("answer", "")

    if answer:
        save_memory(question, answer)
        state.setdefault("steps", []).append("memory: saved this exchange")

    return state


if __name__ == "__main__":
    # Сохраняем тестовую пару
    save_memory(
        "Какая самая массивная чёрная дыра в базе?",
        "Самая массивная чёрная дыра в базе — M87*, её масса около 6.5 млрд солнечных масс.",
    )
    print("Сохранено.")

    # Пробуем вспомнить по похожему вопросу
    results = recall_memory("Расскажи про самую тяжёлую чёрную дыру")
    print(f"\nНайдено {len(results)} воспоминаний:")
    for r in results:
        print("---")
        print(r)