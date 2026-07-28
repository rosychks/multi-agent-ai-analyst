"""
F9 — Graph: собирает supervisor + 4 агентов + critic в единый граф LangGraph.

Логика потока:
1. supervisor решает, куда идти дальше (retriever/web/data_sql/code/finish)
2. после любого специалиста — возврат к supervisor (он решает, нужен ли ещё шаг)
3. когда supervisor решает "finish" — идём в draft_answer, затем critic
4. если critic одобрил (или лимит ревизий исчерпан) — граф завершается
   если не одобрил — возврат в draft_answer для повторной попытки
"""
import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "agents"))

from langgraph.graph import StateGraph, END

from state import AgentState, new_state
from supervisor import supervisor
from critic import draft_answer, critic_agent
from memory import recall_node, save_node
from agents.retriever import retriever_agent
from agents.web import web_agent
from agents.data_sql import data_sql_agent
from agents.code_agent import code_agent


def route_from_supervisor(state: dict) -> str:
    """Читает state['plan'] и решает, в какой узел идти дальше."""
    return state["plan"]


def route_from_critic(state: dict) -> str:
    """Если критик одобрил (или лимит ревизий исчерпан) — конец, иначе снова черновик."""
    if state.get("critic_ok"):
        return "end"
    return "retry"


def build_graph():
    graph = StateGraph(AgentState)

    # --- Узлы ---
    graph.add_node("recall", recall_node)
    graph.add_node("supervisor", supervisor)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("web", web_agent)
    graph.add_node("data_sql", data_sql_agent)
    graph.add_node("code", code_agent)
    graph.add_node("draft_answer", draft_answer)
    graph.add_node("critic", critic_agent)
    graph.add_node("save", save_node)

    # --- Точка входа: сначала вспоминаем релевантную историю ---
    graph.set_entry_point("recall")
    graph.add_edge("recall", "supervisor")

    # --- Супервизор решает, куда идти дальше ---
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "retriever": "retriever",
            "web": "web",
            "data_sql": "data_sql",
            "code": "code",
            "finish": "draft_answer",
        },
    )

    # --- После любого специалиста — снова к супервизору ---
    graph.add_edge("retriever", "supervisor")
    graph.add_edge("web", "supervisor")
    graph.add_edge("data_sql", "supervisor")
    graph.add_edge("code", "supervisor")

    # --- Черновик → критик ---
    graph.add_edge("draft_answer", "critic")

    # --- Критик решает: конец (через сохранение в память) или повторный черновик ---
    graph.add_conditional_edges(
        "critic",
        route_from_critic,
        {
            "end": "save",
            "retry": "draft_answer",
        },
    )
    graph.add_edge("save", END)

    return graph.compile()


# Скомпилированный граф — импортируется другими модулями (frontend, evaluation)
app_graph = build_graph()


def ask(question: str) -> dict:
    """
    Удобная обёртка: задать вопрос и получить финальный state.
    Если Langfuse настроен (F12) — весь прогон графа автоматически
    трейсится: видно каждый узел, время выполнения, токены.
    """
    from config import HAS_LANGFUSE

    initial_state = new_state(question)
    run_config = {"recursion_limit": 25}

    if HAS_LANGFUSE:
        from langfuse.langchain import CallbackHandler

        langfuse_handler = CallbackHandler()
        run_config["callbacks"] = [langfuse_handler]

    final_state = app_graph.invoke(initial_state, config=run_config)
    return final_state


if __name__ == "__main__":
    result = ask("Какая самая массивная чёрная дыра в базе?")

    print("=" * 60)
    print("ВОПРОС:", result["question"])
    print("=" * 60)
    print("ОТВЕТ:", result["answer"])
    print("=" * 60)
    print("Одобрено критиком:", result["critic_ok"])
    print("Ревизий:", result["revisions"])
    print("\nПолный лог шагов:")
    for i, step in enumerate(result["steps"], 1):
        print(f"{i}. {step}")