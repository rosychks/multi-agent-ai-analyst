"""
F7 — Supervisor: LLM со структурированным выводом решает, какого агента
вызвать следующим (retriever / web / data_sql / code / finish), глядя
на вопрос и на то, что уже собрано в state.
"""
import os
import sys

sys.path.append(os.path.dirname(__file__))

from config import llm_strong
from pydantic import BaseModel, Field
from typing import Literal

MAX_STEPS = 6  # защита от бесконечного цикла — жёсткий потолок шагов


class RouteDecision(BaseModel):
    """Структурированный вывод: куда супервизор направляет граф дальше."""
    next_agent: Literal["retriever", "web", "data_sql", "code", "finish"] = Field(
        description="Какой агент должен отработать следующим, либо 'finish' если данных уже достаточно для ответа"
    )
    reasoning: str = Field(description="Короткое объяснение, почему выбран именно этот агент")


# LLM с гарантированным структурированным выводом (Pydantic-схема)
structured_llm = llm_strong.with_structured_output(RouteDecision)


def build_context_summary(state: dict) -> str:
    """Кратко описывает, что уже собрано в state, для промпта супервизора."""
    parts = []
    parts.append(f"Куски из документов (retriever): {'есть, ' + str(len(state.get('documents', []))) + ' шт.' if state.get('documents') else 'нет'}")
    parts.append(f"Веб-результат: {'есть' if state.get('web_result') else 'нет'}")
    parts.append(f"SQL-результат: {'есть' if state.get('sql_result') else 'нет'}")
    parts.append(f"Код-результат: {'есть' if state.get('code_result') else 'нет'}")
    return "\n".join(parts)


def supervisor(state: dict) -> dict:
    """
    Узел графа: решает следующий шаг. Записывает решение в state["plan"].
    Граф (F9) будет читать state["plan"] и направлять выполнение туда.
    """
    question = state["question"]
    steps_done = len(state.get("steps", []))

    # Жёсткая защита от зацикливания — если шагов слишком много, принудительно завершаем
    if steps_done >= MAX_STEPS:
        state["plan"] = "finish"
        state.setdefault("steps", []).append("supervisor: forced finish (max steps reached)")
        return state

    context_summary = build_context_summary(state)

    prompt = f"""Ты — супервизор мульти-агентной системы общего назначения.
Реши, какой агент должен отработать следующим, чтобы ответить на вопрос
пользователя. Система отвечает на ЛЮБЫЕ темы, а не только на астрономию.

Вопрос: "{question}"

Что уже собрано:
{context_summary}

Уже выполненные шаги: {state.get('steps', [])}

Агенты:
- retriever: ищет по загруженной базе документов (в ней сейчас в основном
  материалы про звёзды/чёрные дыры/созвездия) — используй, только если вопрос
  может быть покрыт именно этими документами
- web: ищет свежую информацию в интернете по ЛЮБОЙ теме — используй для любых
  вопросов, где важны актуальные факты, цифры, события или темы, в которых
  ты не уверен на 100%
- data_sql: отвечает на вопросы, требующие точных чисел из базы данных
  (звёзды, чёрные дыры, созвездия) — используй только для вопросов по этой БД
- code: делает расчёты (среднее, сумма и т.д.) по уже найденным данным
- finish: пора формировать финальный ответ — либо данных уже достаточно,
  либо вопрос из области общих знаний, на который ты уверенно ответишь и без
  поиска (например, объяснить общеизвестное понятие, помочь с математикой,
  написать текст и т.п.)

Правила:
- Вопрос НЕ обязан быть про астрономию — отвечай на любые темы
- Если вопрос требует актуальных/точных фактов, которых ты не знаешь наверняка — выбери web, а не сразу finish
- Если вопрос — это общеизвестные знания (объяснение понятия, базовая математика, помощь с текстом) — можно сразу finish, модель ответит сама
- Не вызывай одного и того же агента дважды подряд без необходимости
- Выбери ровно одного агента"""

    decision: RouteDecision = structured_llm.invoke(prompt)

    state["plan"] = decision.next_agent
    state.setdefault("steps", []).append(f"supervisor: routed to '{decision.next_agent}' ({decision.reasoning})")

    return state


if __name__ == "__main__":
    from state import new_state

    test_state = new_state("Какая самая массивная чёрная дыра в базе?")
    result = supervisor(test_state)
    print("Решение:", result["plan"])
    print("Шаги:", result["steps"])

    print("\n--- Второй вызов (представим, что data_sql уже отработал) ---")
    test_state["sql_result"] = "M87*, масса 6.5 млрд солнечных масс"
    result2 = supervisor(test_state)
    print("Решение:", result2["plan"])
    print("Шаги:", result2["steps"])