"""
Вспомогательный скрипт: запускается как ОТДЕЛЬНЫЙ процесс через subprocess.
Читает Python-код из stdin, выполняет его с урезанными builtins,
печатает результат в stdout. Используется code_agent.py для песочницы.
"""
import sys
import io
import contextlib

SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum, "len": len,
    "round": round, "range": range, "sorted": sorted, "enumerate": enumerate,
    "float": float, "int": int, "str": str, "list": list, "dict": dict,
    "tuple": tuple, "zip": zip, "print": print,
}


def main():
    code = sys.stdin.read()
    buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(buffer):
            exec(code, {"__builtins__": SAFE_BUILTINS}, {})
        print("OK", flush=True)
        print(buffer.getvalue(), end="", flush=True)
    except Exception as e:
        print("ERROR", flush=True)
        print(str(e), end="", flush=True)


if __name__ == "__main__":
    main()