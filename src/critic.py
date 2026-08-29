"""
F8 — Draft answer + Critic:
1. draft_answer() — собирает всё найденное (documents, web, sql, code)
   и формирует черновой ответ через LLM. Если данных не найдено — отвечает
   на основе собственных знаний модели (система работает с любыми темами).
   Каждый черновик добавляется в state["draft_history"], чтобы весь цикл
   самопроверки был виден, а не только финальный ответ.
2. critic_agent() — проверяет черновик против собранных данных (если они
   были) или на внутреннюю непротиворечивость (если данных не было).
   Помимо approved/reason, выносит confidence (0-100) — насколько ответ
   подкреплён проверяемыми данными, а не общими знаниями модели.
   Если ответ не подкреплён фактами или неполный — отправляет на
   доработку (увеличивает state["revisions"]), но не бесконечно.
"""
import os
import re
import sys

sys.path.append(os.path.dirname(__file__))

from config import llm, llm_strong
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage

MAX_REVISIONS = 2  # защита от бесконечного цикла критик <-> черновик

NO_EVIDENCE_MARKER = "Данных не найдено."

IDENTITY_SYSTEM_PROMPT = (
    "Тебя зовут Synapse — ИИ-аналитик для анализа данных. Если тебя спросят, "
    "как тебя зовут, кто ты, какая ты модель, кто тебя создал и т.п. — "
    "отвечай, что ты Synapse. Никогда, ни при каких обстоятельствах не "
    "называй себя ChatGPT, GPT, OpenAI, Gemini, Google, Claude, Anthropic "
    "или любым другим названием базовой модели или её создателя — эти "
    "названия должны быть полностью скрыты от пользователя."
)


class CriticVerdict(BaseModel):
    """Структурированный вердикт критика."""
    approved: bool = Field(description="True, если ответ хорошо подкреплён собранными данными и отвечает на вопрос")
    reason: str = Field(description="Короткое объяснение вердикта — что не так, если approved=False")
    confidence: int = Field(
        ge=0,
        le=100,
        description=(
            "0-100: насколько ответ подкреплён проверяемыми данными (документы/веб/SQL/расчёты), "
            "а не общими знаниями модели. Если собранных данных не было вообще — уверенность должна "
            "быть низкой (не выше 40), даже если ответ логически непротиворечив. Если ответ прямо "
            "взят из точных собранных данных — уверенность высокая (85 и выше)."
        ),
    )


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

    return "\n\n".join(parts) if parts else NO_EVIDENCE_MARKER


def draft_answer(state: dict) -> dict:
    """Узел графа: формирует черновой ответ на основе всего собранного.

    Система общего назначения: если найдены данные (документы/веб/БД/расчёты),
    ответ строится в первую очередь на них. Если данных нет (вопрос не по
    астрономической базе — математика, общие знания, помощь с текстом и т.д.),
    модель отвечает на основе собственных знаний, но не выдумывает точные
    факты и цифры, в которых не уверена.
    """
    question = state["question"]
    evidence = build_evidence(state)
    has_evidence = evidence != NO_EVIDENCE_MARKER

    if has_evidence:
        prompt = f"""Ответь на вопрос пользователя. Данные ниже — приоритетный источник:
если они прямо отвечают на вопрос, используй именно их. Если данных недостаточно
для полного ответа — дополни своими знаниями, но не выдумывай цифры и факты,
которых нет ни в данных, ни в твоих проверенных знаниях.

Вопрос: "{question}"

Собранные данные:
{evidence}

Дай короткий, точный, дружелюбный ответ на русском языке."""
    else:
        prompt = f"""Ответь на вопрос пользователя, используя свои знания.
Данных из поиска/базы не найдено — отвечай сам, если уверенно знаешь ответ.
Если вопрос требует актуальных фактов (свежие события, точные цифры), которых
ты не можешь знать наверняка — честно скажи об этом, не выдумывай.

Вопрос: "{question}"

Дай короткий, точный, дружелюбный ответ на русском языке."""

    response = llm.invoke([
        SystemMessage(content=IDENTITY_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])
    answer_text = scrub_identity_leaks(extract_text(response))

    state["answer"] = answer_text
    state.setdefault("steps", []).append("draft_answer: generated")

    # Регистрируем этот черновик в истории — вердикт критика допишется
    # в него же на следующем шаге графа.
    state.setdefault("draft_history", []).append({
        "answer": answer_text,
        "approved": None,
        "reason": None,
        "confidence": None,
    })

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
    draft_history = state.setdefault("draft_history", [])

    if revisions >= MAX_REVISIONS:
        state["critic_ok"] = True
        state["critic_reason"] = "Принудительное одобрение (лимит ревизий исчерпан)."
        if draft_history:
            draft_history[-1]["approved"] = True
            draft_history[-1]["reason"] = state["critic_reason"]
        state.setdefault("steps", []).append("critic: forced approve (max revisions)")
        return state

    evidence_note = evidence if evidence != NO_EVIDENCE_MARKER else (
        "Данных не было — ответ должен опираться на общие знания модели, "
        "без выдумывания точных фактов/цифр, в которых модель не может быть уверена."
    )

    prompt = f"""Ты — критик, проверяющий качество ответа AI-аналитика.

Вопрос: "{question}"

Черновой ответ: "{answer}"

Собранные данные (если есть — ответ должен в первую очередь опираться на них):
{evidence_note}

Проверь:
- Отвечает ли черновик на заданный вопрос?
- Если собранные данные есть — согласуется ли с ними ответ (нет ли выдумок)?
- Если данных не было — не пытается ли модель нафантазировать точные цифры/факты, которых не может знать наверняка?
- Достаточно ли он полный и понятный?
- Оцени confidence: насколько ответ подкреплён именно собранными данными, а не общими знаниями.

Вынеси вердикт."""

    verdict: CriticVerdict = structured_critic_llm.invoke(prompt)

    state["critic_ok"] = verdict.approved
    state["critic_reason"] = verdict.reason
    state["confidence"] = verdict.confidence

    if draft_history:
        draft_history[-1]["approved"] = verdict.approved
        draft_history[-1]["reason"] = verdict.reason
        draft_history[-1]["confidence"] = verdict.confidence

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


_IDENTITY_LEAK_PATTERN = re.compile(
    r"\b(chatgpt|gpt-?\d\S*|openai|gemini|google\s*ai|claude|anthropic)\b",
    re.IGNORECASE,
)


def scrub_identity_leaks(text: str) -> str:
    """Подстраховка: если модель всё же проговорится о реальном провайдере
    или названии базовой модели, подменяем это на 'Synapse', чтобы
    пользователь никогда не увидел настоящее имя модели."""
    return _IDENTITY_LEAK_PATTERN.sub("Synapse", text)


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
    print("Уверенность:", test_state["confidence"])
    print("Ревизий:", test_state["revisions"])
    print("\nИстория черновиков:")
    for i, d in enumerate(test_state["draft_history"], 1):
        print(f"{i}. approved={d['approved']} confidence={d['confidence']} reason={d['reason']}")