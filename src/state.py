"""
F1 — Shared state: единый объект, который течёт через каждый узел графа.
"""
from typing import TypedDict, List, Optional


class AgentState(TypedDict):
    question: str

    plan: str
    steps: List[str]
    revisions: int

    documents: List[str]
    web_result: Optional[str]
    sql_result: Optional[str]
    code_result: Optional[str]

    history: List[str]

    answer: Optional[str]
    critic_ok: Optional[bool]
    critic_reason: Optional[str]


def new_state(question: str) -> AgentState:
    return AgentState(
        question=question,
        plan="",
        steps=[],
        revisions=0,
        documents=[],
        web_result=None,
        sql_result=None,
        code_result=None,
        history=[],
        answer=None,
        critic_ok=None,
        critic_reason=None,
    )


if __name__ == "__main__":
    s = new_state("Сколько клиентов ушло в прошлом квартале?")
    print(s)