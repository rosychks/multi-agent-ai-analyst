"""
F6 — Code agent: LLM пишет короткий Python-скрипт для расчётов
(например, "посчитай среднюю массу"), агент выполняет его в песочнице.

Песочница:
- выполняется в ОТДЕЛЬНОМ процессе (multiprocessing), не в основном
- жёсткий лимит по времени (timeout) — если код завис или считает
  слишком долго, процесс принудительно убивается
- ограниченные builtins — запрещены import os/sys/subprocess/open и т.д.,
  разрешена только простая арифметика/математика
"""
import os
import re
import sys
import subprocess

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from config import llm

TIMEOUT_SECONDS = 5
RUNNER_PATH = os.path.join(os.path.dirname(__file__), "_sandbox_runner.py")

FORBIDDEN_PATTERNS = [
    "import", "open(", "exec(", "eval(", "__", "os.", "sys.",
    "subprocess", "socket", "requests", "input(",
]


def extract_content_text(response) -> str:
    """LLM ответ может быть строкой или списком блоков — приводим к строке."""
    if isinstance(response.content, str):
        return response.content.strip()
    text_parts = [
        block.get("text", "")
        for block in response.content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(text_parts).strip()


def is_safe_code(code: str) -> bool:
    """Простая проверка на запрещённые конструкции перед запуском."""
    lowered = code.lower()
    return not any(pattern in lowered for pattern in FORBIDDEN_PATTERNS)


def generate_code(question: str, context: str = "") -> str:
    """Просит LLM написать короткий Python-скрипт, который печатает ответ."""
    prompt = f"""Ты пишешь короткий Python-скрипт для расчёта.

Контекст (данные, если есть): {context}

Задача: "{question}"

Правила:
- НЕ используй import, open, exec, eval — только чистые вычисления
- В конце обязательно сделай print(результат)
- Верни ТОЛЬКО код, без пояснений, без markdown, без ```."""

    response = llm.invoke(prompt)
    code = extract_content_text(response)

    code = re.sub(r"^```python\s*|\s*```$", "", code, flags=re.IGNORECASE).strip()
    code = re.sub(r"^```\s*|\s*```$", "", code).strip()

    return code


def run_in_sandbox(code: str, timeout: int = TIMEOUT_SECONDS):
    """
    Запускает код в отдельном процессе (subprocess) с жёстким timeout.
    Используется отдельный .py-скрипт (_sandbox_runner.py) вместо
    multiprocessing — надёжнее работает на Windows/в Jupyter.
    """
    try:
        result = subprocess.run(
            [sys.executable, RUNNER_PATH],
            input=code,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return "error", f"Превышен лимит времени ({timeout} сек), процесс остановлен."

    output = result.stdout
    if output.startswith("OK\n"):
        return "ok", output[3:]
    elif output.startswith("ERROR\n"):
        return "error", output[6:]
    else:
        # Что-то пошло совсем не так (например, синтаксическая ошибка на старте)
        return "error", result.stderr or output or "Неизвестная ошибка выполнения."


def code_agent(state: dict) -> dict:
    """Узел графа: пишет и безопасно выполняет Python-код для расчётов."""
    question = state["question"]
    context = "\n".join(state.get("documents", []))[:1000]

    code = generate_code(question, context)

    if not is_safe_code(code):
        state["code_result"] = f"Код отклонён (запрещённые конструкции):\n{code}"
        state.setdefault("steps", []).append("code: rejected unsafe code")
        return state

    status, output = run_in_sandbox(code)

    if status == "ok":
        state["code_result"] = f"Код:\n{code}\n\nВывод:\n{output}"
        state.setdefault("steps", []).append("code: executed successfully")
    else:
        state["code_result"] = f"Ошибка выполнения: {output}\nКод был:\n{code}"
        state.setdefault("steps", []).append(f"code: error - {output}")

    return state


if __name__ == "__main__":
    from state import new_state

    # На Windows multiprocessing требует запуска через if __name__ == "__main__"
    test_state = new_state("Посчитай среднее арифметическое чисел 4, 8.6, 21, 5.4, 12")
    result = code_agent(test_state)

    print(result["code_result"])
    print("\nШаги:", result["steps"])

    print("\n--- Проверка guard: попытка import os ---")
    dangerous_code = "import os\nprint(os.listdir('.'))"
    print("safe?", is_safe_code(dangerous_code))

    print("\n--- Проверка timeout: бесконечный цикл ---")
    status, output = run_in_sandbox("while True: pass", timeout=3)
    print(status, output)