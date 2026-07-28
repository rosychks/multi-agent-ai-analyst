"""
F13 (backend half) — FastAPI сервер, который стримит выполнение графа
в реальном времени через Server-Sent Events (SSE). Фронтенд (Next.js)
подключается к /ask и видит каждый шаг агентов по мере выполнения,
а не ждёт полного ответа молча.
"""
import os
import sys
import json

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from graph import app_graph
from state import new_state

app = FastAPI(title="Multi-Agent AI Analyst API")

# Разрешаем запросы с фронтенда (Next.js обычно на localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str


def format_sse(event_type: str, data: dict) -> str:
    """Формирует строку в формате Server-Sent Events."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n"


def stream_graph_events(question: str):
    """
    Генератор: прогоняет граф через .stream() (не .invoke()) —
    LangGraph отдаёт state ПОСЛЕ каждого узла, а не только в конце.
    Каждое обновление отправляем на фронтенд отдельным SSE-событием.
    """
    initial_state = new_state(question)

    try:
        for chunk in app_graph.stream(initial_state, config={"recursion_limit": 25}):
            # chunk — это {"имя_узла": обновлённый_state}
            for node_name, node_state in chunk.items():
                yield format_sse("step", {
                    "node": node_name,
                    "steps_log": node_state.get("steps", []),
                })

                # Если это финальный узел (save) — отправим итоговый ответ отдельно
                if node_name == "save":
                    yield format_sse("final", {
                        "answer": node_state.get("answer", ""),
                        "critic_ok": node_state.get("critic_ok"),
                        "revisions": node_state.get("revisions"),
                        "steps": node_state.get("steps", []),
                    })
    except Exception as e:
        yield format_sse("error", {"message": str(e)})


@app.post("/ask")
async def ask_endpoint(request: AskRequest):
    """SSE-эндпоинт: стримит шаги графа по мере выполнения."""
    return StreamingResponse(
        stream_graph_events(request.question),
        media_type="text/event-stream",
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)