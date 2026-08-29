"""
F1 — Config: читает ключи из .env и создаёт готовые к использованию
клиенты (LLM, эмбеддинги, Qdrant). Все остальные модули импортируют
объекты отсюда, а не читают .env заново.

Вызовы к Gemini идут через официальный коннектор langchain-google-genai
(родной API Google), а не через OpenAI-совместимый прокси/эндпоинт —
это надёжнее и полностью поддерживает структурированный вывод
(with_structured_output), который используют supervisor и critic.
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
        "в корне проекта и содержит GEMINI_API_KEY=... (ваш личный ключ "
        "с https://aistudio.google.com/apikey)."
    )

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)

llm_strong = llm

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GEMINI_API_KEY,
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