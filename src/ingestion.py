"""
F2 — Ingestion: читает PDF, режет на куски (chunks), считает эмбеддинги
через Gemini и загружает всё в коллекцию Qdrant.
Запускать один раз (или заново при обновлении документов).
"""
import os
import sys

sys.path.append(os.path.dirname(__file__))  # чтобы найти config.py рядом

from config import embeddings, qdrant_client
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import VectorParams, Distance, PointStruct

COLLECTION_NAME = "documents"
PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "documents")


def load_and_split(pdf_folder: str):
    """Загружает все PDF из папки и режет на куски по ~1000 символов."""
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

    for filename in os.listdir(pdf_folder):
        if not filename.lower().endswith(".pdf"):
            continue
        full_path = os.path.join(pdf_folder, filename)
        loader = PyPDFLoader(full_path)
        pages = loader.load()
        chunks = splitter.split_documents(pages)
        print(f"{filename}: {len(pages)} страниц → {len(chunks)} кусков")
        all_chunks.extend(chunks)

    return all_chunks


def ingest():
    chunks = load_and_split(PDF_PATH)
    if not chunks:
        print("PDF не найдены в", PDF_PATH)
        return

    sample_vector = embeddings.embed_query(chunks[0].page_content)
    vector_size = len(sample_vector)

    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    texts = [c.page_content for c in chunks]
    vectors = embeddings.embed_documents(texts)

    points = [
        PointStruct(
            id=i,
            vector=vectors[i],
            payload={
                "text": texts[i],
                "source": chunks[i].metadata.get("source", ""),
                "page": chunks[i].metadata.get("page", None),
            },
        )
        for i in range(len(chunks))
    ]

    qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Загружено {len(points)} кусков в коллекцию '{COLLECTION_NAME}'.")


def test_search(query: str, k: int = 3):
    """Быстрая проверка similarity search."""
    query_vector = embeddings.embed_query(query)
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
    )
    for r in results.points:
        print(f"score={r.score:.3f} | {r.payload['text'][:120]}...")


if __name__ == "__main__":
    ingest()
    print("\n--- Тестовый поиск ---")
    test_search("о чём этот документ?")