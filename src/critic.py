"""
F8 — Draft answer + Critic:
1. draft_answer() — собирает всё найденное (documents, web, sql, code)
   и формирует черновой ответ через LLM.
2. critic_agent() — проверяет черновик против собранных данных.
   Если ответ не подкреплён фактами или неполный — отправляет на
   доработку (увеличивает state["revisions"]), но не бесконечно.
"""
import os
import sys

sys.path.append(os.path.dirname(__file__))

from config import llm, llm_strong
from pydantic import BaseModel, Field

MAX_REVISIONS = 2  # защита от бесконечного цикла критик <-> черновик


class CriticVerdict(BaseModel):
    """Структурированный вердикт критика."""
    approved: bool = Field(description="True, если ответ хорошо подкреплён собранными данными и отвечает на вопрос")
    reason: str = Field(description="Короткое объяснение вердикта — что не так, если approved=False")


structured_critic_llm = llm_strong.with_structured_output(CriticVerdict)


def build_evidence(state: dict) -> str:
    """Собирает все найденные данные в один текстовый блок для промпта."""
    parts = []
    if state.get("documents"):
        parts.append("=== Из документов ===\n" + "\n---\n".join(state["documents"]))
    if state.get("web_result"):
        parts.append("=== Из веб-поиска ===\n" + state["web_result"])
    if state.get("sql_result"):
        parts.append("=== Из базы данных ===\n" + state["sql_result"])
    if state.get("code_result"):
        parts.append("=== Из расчётов ===\n" + state["code_result"])

    return "\n\n".join(parts) if parts else "Данных не найдено."


def draft_answer(state: dict) -> dict:
    """Узел графа: формирует черновой ответ на основе всего собранного."""
    question = state["question"]
    evidence = build_evidence(state)

    prompt = f"""Ответь на вопрос пользователя, опираясь ТОЛЬКО на данные ниже.
Если данных не хватает — честно скажи об этом, не выдумывай.

Вопрос: "{question}"

Собранные данные:
{evidence}

Дай короткий, точный, дружелюбный ответ на русском языке."""

    response = llm.invoke(prompt)
    answer_text = extract_text(response)

    state["answer"] = answer_text
    state.setdefault("steps", []).append("draft_answer: generated")

    return state


def critic_agent(state: dict) -> dict:
    """
    Узел графа: проверяет state["answer"] против собранных данных.
    Если отклонён и лимит ревизий не исчерпан — увеличивает revisions
    и НЕ одобряет (граф в F9 отправит на повторный draft_answer).
    Если лимит исчерпан — принудительно одобряет, чтобы не зациклиться.
    """
    question = state["question"]
    answer = state.get("answer", "")
    evidence = build_evidence(state)
    revisions = state.get("revisions", 0)

    if revisions >= MAX_REVISIONS:
        state["critic_ok"] = True
        state["critic_reason"] = "Принудительное одобрение (лимит ревизий исчерпан)."
        state.setdefault("steps", []).append("critic: forced approve (max revisions)")
        return state

    prompt = f"""Ты — критик, проверяющий качество ответа AI-аналитика.

Вопрос: "{question}"

Черновой ответ: "{answer}"

Собранные данные (то, на чём должен основываться ответ):
{evidence}

Проверь:
- Отвечает ли черновик на заданный вопрос?
- Подкреплён ли он собранными данными (нет ли выдумок)?
- Достаточно ли он полный и понятный?

Вынеси вердикт."""

    verdict: CriticVerdict = structured_critic_llm.invoke(prompt)

    state["critic_ok"] = verdict.approved
    state["critic_reason"] = verdict.reason

    if not verdict.approved:
        state["revisions"] = revisions + 1
        state.setdefault("steps", []).append(f"critic: rejected ({verdict.reason})")
    else:
        state.setdefault("steps", []).append("critic: approved")

    return state


def extract_text(response) -> str:
    """LLM ответ может быть строкой или списком блоков — приводим к строке."""
    if isinstance(response.content, str):
        return response.content.strip()
    text_parts = [
        block.get("text", "")
        for block in response.content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(text_parts).strip()


if __name__ == "__main__":
    from state import new_state

    test_state = new_state("Какая самая массивная чёрная дыра в базе?")
    test_state["sql_result"] = (
        "SQL: SELECT * FROM black_holes ORDER BY mass_solar DESC LIMIT 1;\n"
        "Результат:\n{'id': 5, 'name': 'M87*', 'location': 'Virgo', "
        "'distance_light_years': 55000000.0, 'mass_solar': 6500000000.0}"
    )

    test_state = draft_answer(test_state)
    print("Черновик:", test_state["answer"])

    test_state = critic_agent(test_state)
    print("\nОдобрено:", test_state["critic_ok"])
    print("Причина:", test_state["critic_reason"])
    print("Ревизий:", test_state["revisions"])