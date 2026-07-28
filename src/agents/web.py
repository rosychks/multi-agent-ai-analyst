"""
F4 — Web agent: ищет свежую информацию в интернете через Tavily.
Работает "gracefully" без ключа — если TAVILY_API_KEY не задан,
просто помечает, что веб-поиск недоступен, не ломая весь граф.
"""
import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import TAVILY_API_KEY, HAS_TAVILY


def web_agent(state: dict, max_results: int = 3) -> dict:
    """
    Ищет вопрос в интернете через Tavily и кладёт краткую сводку
    в state["web_result"]. Если ключа нет — не падает, а честно
    сообщает об отсутствии веб-поиска.
    """
    question = state["question"]

    if not HAS_TAVILY:
        state["web_result"] = "Веб-поиск недоступен (TAVILY_API_KEY не задан)."
        state.setdefault("steps", []).append("web: skipped (no API key)")
        return state

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=question, max_results=max_results)

        # Собираем краткую сводку из найденных источников
        snippets = []
        for r in response.get("results", []):
            title = r.get("title", "")
            content = r.get("content", "")[:300]
            url = r.get("url", "")
            snippets.append(f"- {title}: {content}... (источник: {url})")

        summary = "\n".join(snippets) if snippets else "Ничего не найдено."
        state["web_result"] = summary
        state.setdefault("steps", []).append(f"web: found {len(snippets)} results")

    except Exception as e:
        # Любая ошибка сети/API не должна ронять весь граф
        state["web_result"] = f"Ошибка веб-поиска: {e}"
        state.setdefault("steps", []).append(f"web: error - {e}")

    return state


if __name__ == "__main__":
    from state import new_state

    test_state = new_state("Какие последние открытия про чёрные дыры в 2026 году?")
    result = web_agent(test_state)

    print("Веб-результат:\n", result["web_result"])
    print("\nШаги:", result["steps"])