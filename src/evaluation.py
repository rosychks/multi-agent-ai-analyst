"""
F11 — Evaluation: прогоняет тестовый набор из 10+ вопросов через граф
и оценивает качество ответов тремя метриками в стиле RAGAS:

- faithfulness       — не выдумал ли ответ факты, которых нет в собранных данных
- answer_relevancy    — отвечает ли ответ на заданный вопрос
- context_precision   — насколько собранные данные (documents/sql/web/code)
                        релевантны вопросу (не мусор ли нашли агенты)

Метрики реализованы через LLM-judge (структурированный вывод), а не через
библиотеку ragas напрямую — так меньше хрупких зависимостей и лишних
вызовов API, при этом смысл метрик тот же, что в RAGAS.
"""
import os
import sys
import time
import json

sys.path.append(os.path.dirname(__file__))

from config import llm
from graph import ask
from pydantic import BaseModel, Field

# --- Тестовый набор: 10+ вопросов, покрывающих всех агентов ---
TEST_QUESTIONS = [
    "Какая самая массивная чёрная дыра в базе?",
    "Сколько световых лет до Sagittarius A*?",
    "Какая звезда ближе всего к Земле по данным базы?",
    "В каком созвездии находится больше всего звёзд?",
    "Расскажи про чёрные дыры звёздной массы из документа.",
    "Какие созвездия лучше всего видны в июле?",
    "Посчитай среднюю массу всех чёрных дыр в базе.",
    "Какие последние новости про открытия в астрономии в 2026 году?",
    "Что такое красный сверхгигант?",
    "Сколько звёзд-красных гигантов в базе данных?",
    "Какая разница в массе между Sirius и Rigel?",
    "Опиши созвездие Орион по данным из документа.",
]


class MetricScore(BaseModel):
    score: float = Field(description="Оценка от 0.0 до 1.0")
    reasoning: str = Field(description="Короткое обоснование оценки")


scoring_llm = llm.with_structured_output(MetricScore)


def score_faithfulness(answer: str, evidence: str) -> MetricScore:
    prompt = f"""Оцени faithfulness (правдивость) ответа: насколько все факты
в ответе подкреплены собранными данными, без выдумок.

Собранные данные:
{evidence[:2000]}

Ответ: "{answer}"

Оценка 1.0 = всё подкреплено данными, 0.0 = ответ полностью выдуман."""
    return scoring_llm.invoke(prompt)


def score_answer_relevancy(question: str, answer: str) -> MetricScore:
    prompt = f"""Оцени answer relevancy: насколько ответ действительно
отвечает на заданный вопрос (не уходит в сторону, не общие слова).

Вопрос: "{question}"
Ответ: "{answer}"

Оценка 1.0 = точно отвечает на вопрос, 0.0 = вообще не по теме."""
    return scoring_llm.invoke(prompt)


def score_context_precision(question: str, evidence: str) -> MetricScore:
    prompt = f"""Оцени context precision: насколько собранные данные
(контекст, найденный агентами) релевантны вопросу, без лишнего мусора.

Вопрос: "{question}"
Собранные данные:
{evidence[:2000]}

Оценка 1.0 = все данные точно по теме, 0.0 = данные не относятся к вопросу."""
    return scoring_llm.invoke(prompt)


def build_evidence_text(state: dict) -> str:
    parts = []
    if state.get("documents"):
        parts.append("\n---\n".join(state["documents"]))
    if state.get("web_result"):
        parts.append(state["web_result"])
    if state.get("sql_result"):
        parts.append(state["sql_result"])
    if state.get("code_result"):
        parts.append(state["code_result"])
    return "\n\n".join(parts) if parts else ""


def evaluate_single(question: str) -> dict:
    """Прогоняет один вопрос через граф и оценивает тремя метриками."""
    result = ask(question)
    evidence = build_evidence_text(result)
    answer = result.get("answer", "")

    faithfulness = score_faithfulness(answer, evidence)
    relevancy = score_answer_relevancy(question, answer)
    precision = score_context_precision(question, evidence)

    return {
        "question": question,
        "answer": answer,
        "critic_ok": result.get("critic_ok"),
        "revisions": result.get("revisions"),
        "faithfulness": faithfulness.score,
        "faithfulness_reason": faithfulness.reasoning,
        "answer_relevancy": relevancy.score,
        "relevancy_reason": relevancy.reasoning,
        "context_precision": precision.score,
        "precision_reason": precision.reasoning,
    }


def run_evaluation(questions: list = None, delay_seconds: float = 2.0) -> list:
    """
    Прогоняет весь тестовый набор. delay_seconds — пауза между вопросами,
    чтобы не упереться в лимиты API за один прогон (каждый вопрос — это
    5-10+ вызовов LLM: supervisor несколько раз, агенты, critic, 3 judge-метрики).
    """
    questions = questions or TEST_QUESTIONS
    results = []

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q}")
        try:
            r = evaluate_single(q)
            results.append(r)
            print(
                f"  faithfulness={r['faithfulness']:.2f}  "
                f"relevancy={r['answer_relevancy']:.2f}  "
                f"precision={r['context_precision']:.2f}  "
                f"critic_ok={r['critic_ok']}"
            )
        except Exception as e:
            print(f"  ОШИБКА: {e}")
            results.append({"question": q, "error": str(e)})

        time.sleep(delay_seconds)

    return results


def summarize(results: list):
    """Печатает средние значения метрик по всему набору."""
    valid = [r for r in results if "error" not in r]
    if not valid:
        print("Нет валидных результатов.")
        return

    avg_faith = sum(r["faithfulness"] for r in valid) / len(valid)
    avg_rel = sum(r["answer_relevancy"] for r in valid) / len(valid)
    avg_prec = sum(r["context_precision"] for r in valid) / len(valid)
    approved_rate = sum(1 for r in valid if r["critic_ok"]) / len(valid)

    print("\n" + "=" * 50)
    print(f"Вопросов оценено: {len(valid)} / {len(results)}")
    print(f"Средний faithfulness:      {avg_faith:.2f}")
    print(f"Средний answer_relevancy:  {avg_rel:.2f}")
    print(f"Средний context_precision: {avg_prec:.2f}")
    print(f"Доля одобренных критиком:  {approved_rate:.0%}")
    print("=" * 50)


if __name__ == "__main__":
    results = run_evaluation()
    summarize(results)

    # Сохраняем полные результаты в файл — пригодится для README/отчёта
    output_path = os.path.join(os.path.dirname(__file__), "..", "evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nПолные результаты сохранены в {output_path}")