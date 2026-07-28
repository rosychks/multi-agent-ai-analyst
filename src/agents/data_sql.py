"""
F5 — SQL/Data agent: по вопросу пользователя LLM пишет SQL-запрос,
агент выполняет его на базе astro.db СТРОГО в режиме "только чтение".

Защита (read-only guard):
- запрещены любые запросы, кроме SELECT
- запрещены ключевые слова DROP/DELETE/UPDATE/INSERT/ALTER/ATTACH/PRAGMA
- соединение с базой открывается в режиме read-only на уровне SQLite (?mode=ro)
"""
import os
import re
import sqlite3
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import llm

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "astro.db")

FORBIDDEN_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
    "ATTACH", "PRAGMA", "CREATE", "REPLACE", "TRUNCATE",
]

SCHEMA_DESCRIPTION = """
Таблица stars (звёзды):
- id, name, constellation, distance_light_years, mass_solar, star_type

Таблица black_holes (чёрные дыры):
- id, name, location, distance_light_years, mass_solar, discovery_year

Таблица constellations (созвездия):
- id, name, hemisphere, best_viewing_month, star_count
"""


def is_safe_sql(query: str) -> bool:
    """Пропускает только одиночный SELECT-запрос без запрещённых ключевых слов."""
    normalized = query.strip().upper()
    if not normalized.startswith("SELECT"):
        return False
    if ";" in query.strip().rstrip(";"):
        # запрещаем несколько запросов через точку с запятой (SQL injection risk)
        return False
    for word in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{word}\b", normalized):
            return False
    return True


def generate_sql(question: str) -> str:
    """Просит LLM написать SQL-запрос по схеме базы."""
    prompt = f"""Ты — генератор SQL-запросов для SQLite базы данных.

Схема базы:
{SCHEMA_DESCRIPTION}

Вопрос пользователя: "{question}"

Напиши ОДИН SELECT-запрос, который отвечает на вопрос.
Верни ТОЛЬКО сам SQL-запрос, без пояснений, без markdown, без ```."""

    response = llm.invoke(prompt)

    # content может быть строкой ИЛИ списком блоков — обрабатываем оба случая
    if isinstance(response.content, str):
        sql = response.content.strip()
    else:
        # список блоков вида [{'type': 'text', 'text': '...'}]
        text_parts = [
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        sql = "".join(text_parts).strip()

    # На случай если модель всё же обернула в markdown-блок
    sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"^```\s*|\s*```$", "", sql).strip()

    return sql


def run_readonly_query(sql: str):
    """Выполняет SQL строго в режиме read-only."""
    db_uri = f"file:{os.path.abspath(DB_PATH)}?mode=ro"
    conn = sqlite3.connect(db_uri, uri=True)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        columns = [desc[0] for desc in cur.description] if cur.description else []
        rows = cur.fetchall()
        return columns, rows
    finally:
        conn.close()


def data_sql_agent(state: dict) -> dict:
    """
    Узел графа: генерирует SQL по вопросу, проверяет на безопасность,
    выполняет read-only и кладёт результат в state["sql_result"].
    """
    question = state["question"]

    sql = generate_sql(question)

    if not is_safe_sql(sql):
        state["sql_result"] = f"Запрос отклонён (не прошёл проверку безопасности): {sql}"
        state.setdefault("steps", []).append("data_sql: rejected unsafe query")
        return state

    try:
        columns, rows = run_readonly_query(sql)
        if not rows:
            result_text = f"SQL: {sql}\nРезультат: пусто."
        else:
            preview = "\n".join(str(dict(zip(columns, row))) for row in rows[:10])
            result_text = f"SQL: {sql}\nРезультат:\n{preview}"
        state["sql_result"] = result_text
        state.setdefault("steps", []).append(f"data_sql: {len(rows)} rows")
    except Exception as e:
        state["sql_result"] = f"Ошибка выполнения SQL: {e}\nЗапрос был: {sql}"
        state.setdefault("steps", []).append(f"data_sql: error - {e}")

    return state


if __name__ == "__main__":
    from state import new_state

    test_state = new_state("Какая самая массивная чёрная дыра в базе?")
    result = data_sql_agent(test_state)

    print(result["sql_result"])
    print("\nШаги:", result["steps"])

    # Проверка, что guard реально блокирует опасные запросы
    print("\n--- Проверка guard ---")
    print("DROP TABLE stars;", "-> safe?", is_safe_sql("DROP TABLE stars;"))
    print("DELETE FROM stars;", "-> safe?", is_safe_sql("DELETE FROM stars;"))
    print("SELECT * FROM stars;", "-> safe?", is_safe_sql("SELECT * FROM stars;"))