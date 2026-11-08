"""Record lecture 3's actual CLI answers and traces without storing API keys.

Run from the repository root:
    uv run --env-file .env python -m scripts.record_lec03 --condition exercise
    uv run --env-file .env python -m scripts.record_lec03 --condition baseline
Then delete SEARCH_RULE from app/agent.py and run --condition without_search_rule.
Also remove the search tool's instruction to use it before answering to run
--condition without_search_prompts. Restore both texts after the experiment.
Existing results are never overwritten.
"""

import argparse
import ast
import hashlib
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from app import agent, tools
from app.config import MODEL, get_api_key

ROOT = Path(__file__).resolve().parent.parent
SEARCH_RULE = "- 用語や講座に固有のことを答える前に、必ず search_glossary で用語集を確認する。\n"
QUESTIONS = [
    ("homework_levels", "AIE2026の講座の宿題にある「3レベル」とは何ですか？"),
    ("lora_qlora", "LoRAとQLoRAの違いは？"),
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True,
                        choices=["exercise", "baseline", "without_search_rule",
                                 "without_search_prompts"])
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    no_global_rule = args.condition in {"without_search_rule", "without_search_prompts"}
    assert (SEARCH_RULE in agent.INSTRUCTIONS) == (not no_global_rule)
    assert ("答える前に使うこと。" in inspect.getdoc(tools.search_glossary)) == (
        args.condition != "without_search_prompts"
    )
    api_key = get_api_key()
    out = ROOT / "lectures/03/results" / args.condition
    out.mkdir(parents=True, exist_ok=False)

    def save(name, value):
        text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
        (out / name).write_text(text.replace(api_key, "[REDACTED]") + "\n", encoding="utf-8")

    source_paths = ["app/agent.py", "app/tools.py", "app/ask.py", "app/config.py",
                    "data/glossary.json", "pyproject.toml", "uv.lock", "scripts/record_lec03.py"]
    metadata = {
        "condition": args.condition, "started_at_utc": utc_now(), "model": MODEL,
        "python": platform.python_version(), "platform": platform.platform(),
        "versions": {p: version(p) for p in ["pydantic-ai-slim", "google-genai"]},
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "instructions": agent.INSTRUCTIONS,
        "tool_descriptions": {f.__name__: inspect.getdoc(f) for f in
                              [tools.search_glossary, tools.get_lecture_summary, tools.make_quiz]},
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths},
        "model_settings": "course CLI defaults; no temperature or seed override",
        "conversation": "fresh process, no conversation history for every run",
        "assisted_by": "Codex", "status": "running",
    }
    save("metadata.json", metadata)
    if args.condition == "exercise":
        jobs = [("plain_llm", "app.ask", QUESTIONS[0][1], 1),
                ("make_quiz", "app.agent", "第2回の内容から確認問題を1問出して", 1)]
    else:
        jobs = [(name, "app.agent", question, repeat)
                for repeat in range(1, args.repetitions + 1) for name, question in QUESTIONS]
    results = []
    for name, module, question, repeat in jobs:
        command = ["-m", module] + (["--trace"] if module == "app.agent" else []) + [question]
        started = utc_now()
        t0 = time.monotonic()
        process = subprocess.run([sys.executable, *command], cwd=ROOT,
                                 env=os.environ | {"PYDANTIC_AI_NO_BANNER": "1"},
                                 capture_output=True, text=True, timeout=180)
        stdout = process.stdout.replace(api_key, "[REDACTED]")
        stderr = process.stderr.replace(api_key, "[REDACTED]")
        answer, _, trace = stdout.partition("\n--- trace ---\n")
        calls = []
        for tool_name, raw_args in re.findall(r"^\[tool call\]\s+(\w+)\((.*)\)$", trace, re.M):
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                arguments = ast.literal_eval(raw_args)
            calls.append({"name": tool_name, "arguments": arguments})
        usage = next((line for line in trace.splitlines() if line.startswith("[usage]")), None)
        requests = re.search(r"\brequests=(\d+)", usage or "")
        row = {
            "question_id": name, "question": question, "repeat": repeat,
            "command": ["python", *command], "started_at_utc": started,
            "finished_at_utc": utc_now(), "seconds": round(time.monotonic() - t0, 3),
            "returncode": process.returncode, "stdout": stdout, "stderr": stderr,
            "answer": answer.removeprefix(f"[model] {MODEL}\n").strip(),
            "tool_calls": calls, "usage": usage,
            "requests": int(requests.group(1)) if requests else None,
        }
        save(f"{name}_{repeat}.json", row)
        results.append(row)
        save("runs.json", results)
        print(json.dumps({"condition": args.condition, "question": name, "repeat": repeat,
                          "returncode": process.returncode, "tool_calls": calls,
                          "requests": row["requests"]}, ensure_ascii=False), flush=True)
        if process.returncode != 0 or (module == "app.agent" and row["requests"] is None):
            metadata["status"] = "failed"
            save("metadata.json", metadata)
            raise SystemExit("CLI failed or trace could not be parsed; inspect the saved record.")
    metadata.update(status="completed", finished_at_utc=utc_now(), run_count=len(results))
    save("metadata.json", metadata)


if __name__ == "__main__":
    main()
