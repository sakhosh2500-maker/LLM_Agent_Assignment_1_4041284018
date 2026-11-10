import argparse
import json
import os
import re
import subprocess
import sys
import traceback

import ollama

MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")

CODE_PROMPT = (
    "Write a SHORT but CORRECT Python 3 program that solves the problem below by "
    "computation (use brute force / enumeration / simple formulas when helpful). "
    "Read every constraint exactly. Be careful with: 0-based indexing (the last "
    "element is index -1), direction ('A immediately after C' means "
    "pos(A) == pos(C) + 1, not just adjacent), percentages, and counting from 0. "
    "The program must print ONLY the final answer, exactly in the format the "
    "problem asks for. Reply with only one ```python code block, no explanation."
)

DIRECT_PROMPT = "\n\nReply with ONLY the final answer, nothing else."


def log(*parts):
    with open("last_run.log", "a", encoding="utf-8") as f:
        f.write(" ".join(str(p) for p in parts) + "\n")


def chat(messages, max_tokens):
    r = ollama.chat(
        model=MODEL,
        messages=messages,
        think=False,
        options={"temperature": 0, "num_ctx": 2048, "num_predict": max_tokens},
    )
    return r["message"]["content"]


def get_code(text):
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(?:python)?\s*(.*)", text, re.S)  # cut-off block
    return m.group(1).strip() if m else ""


def run_code(code):
    """Run the program. Returns (answer, error_message)."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    try:
        p = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=20,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return "", "The program ran too long (timeout)."
    out = p.stdout.strip()
    if p.returncode != 0:
        return "", (p.stderr.strip().splitlines() or ["error"])[-1]
    if not out:
        return "", "The program printed nothing (no valid solution found)."
    return out.splitlines()[-1].strip(), ""


def solve(problem):
    messages = [
        {"role": "system", "content": CODE_PROMPT},
        {"role": "user", "content": problem},
    ]
    # Steps 1-2: write a program, run it; if it fails, show the error and ask for a fix
    for attempt in range(2):
        try:
            raw = chat(messages, 450)
            code = get_code(raw)
            log(f"=== CODE (attempt {attempt + 1}) ===", code)
            if not code:
                break
            ans, err = run_code(code)
            log("=== PROGRAM OUTPUT ===", repr(ans), "| error:", err)
            if ans:
                return ans
            messages += [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": "Your program failed: " + err + " Re-read every "
                    "constraint, fix the bug, and reply with only the corrected "
                    "```python block.",
                },
            ]
        except Exception:
            traceback.print_exc(file=sys.stderr)
            break
    # Step 3 (fallback): ask directly for the final answer
    try:
        log("=== FALLBACK (direct answer) ===")
        short = chat([{"role": "user", "content": problem + DIRECT_PROMPT}], 40)
        return short.strip().splitlines()[-1].strip() if short.strip() else ""
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    with open(a.input, encoding="utf-8") as f:
        data = json.load(f)

    answer = solve(data["problem"])
    print("ANSWER:", repr(answer), file=sys.stderr)

    with open(a.output, "w", encoding="utf-8") as f:
        json.dump({"id": data["id"], "answer": answer}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
