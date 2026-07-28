"""
F1 — Config: читает ключи из .env и создаёт готовые к использованию
клиенты (LLM, эмбеддинги, Qdrant). Все остальные модули импортируют
объекты отсюда, а не читают .env заново.

Все вызовы к Gemini идут через курсовой прокси (OpenAI-совместимый API),
а не напрямую в Google — это решает проблему с личными дневными лимитами.
"""
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY не найден. Проверьте, что файл .env лежит "
        "в корне проекта и содержит GEMINI_API_KEY=... (курсовой прокси-ключ)."
    )

PROXY_BASE_URL = "https://saidazam-litellm-proxy.hf.space/v1"

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = ChatOpenAI(
    base_url=PROXY_BASE_URL,
    api_key=GEMINI_API_KEY,
    model="gemini-flash-lite",
    temperature=0,
)

llm_strong = llm

embeddings = OpenAIEmbeddings(
    base_url=PROXY_BASE_URL,
    api_key=GEMINI_API_KEY,
    model="gemini-embedding",
)

from qdrant_client import QdrantClient

if QDRANT_URL and QDRANT_API_KEY:
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
else:
    qdrant_client = QdrantClient(path="./qdrant_data")

HAS_TAVILY = bool(TAVILY_API_KEY)
HAS_LANGFUSE = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)


if __name__ == "__main__":
    print("GEMINI_API_KEY:", "OK" if GEMINI_API_KEY else "MISSING")
    print("QDRANT:", "cloud" if (QDRANT_URL and QDRANT_API_KEY) else "embedded (local)")
    print("Tavily:", "enabled" if HAS_TAVILY else "disabled (optional)")
    print("Langfuse:", "enabled" if HAS_LANGFUSE else "disabled (optional)")

    resp = llm.invoke("Скажи одно слово: работает")
    print("LLM (flash-lite) test response:", resp.content)

    resp2 = llm_strong.invoke("Скажи одно слово: работает")
    print("LLM (flash, supervisor/critic) test response:", resp2.content)

    print("Qdrant collections:", qdrant_client.get_collections())