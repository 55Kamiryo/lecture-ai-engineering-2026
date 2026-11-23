"""第5回のCLI実行結果と、取得したページの記録を保存する。

例: uv run --env-file .env python -m scripts.record_lec05 \
    --output lectures/05/results/vector_k5_lora -- --trace 'LoRAとQLoRAの違いは？'
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys

from pydantic_ai.messages import ToolCallPart, ToolReturnPart

from app import agent, config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("agent_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if any(args.output.with_suffix(ext).exists() for ext in (".log", ".json")):
        parser.error("既存の実行記録があります。別の--outputを指定してください")
    cli_args = args.agent_args
    if cli_args[:1] == ["--"]:
        cli_args = cli_args[1:]
    if "--trace" not in cli_args:
        parser.error("app.agentに渡す引数に--traceを指定してください")

    record = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "agent_args": cli_args,
        "model": config.MODEL,
        "rag_top_k": config.RAG_TOP_K,
        "source_sha256": sha256(config.RAG_SOURCE.read_bytes()).hexdigest(),
        "index_sha256": sha256(config.RAG_DB.read_bytes()).hexdigest(),
    }
    original_trace = agent.print_trace

    def save_trace(result, elapsed_seconds=None):
        events = []
        for message in result.all_messages():
            for part in message.parts:
                if isinstance(part, ToolCallPart):
                    events.append({"type": "call", "tool": part.tool_name,
                                   "id": part.tool_call_id, "args": part.args_as_dict()})
                elif isinstance(part, ToolReturnPart):
                    content = part.content
                    if isinstance(content, str):
                        try:
                            content = json.loads(content)
                        except json.JSONDecodeError:
                            pass
                    events.append({"type": "return", "tool": part.tool_name,
                                   "id": part.tool_call_id, "content": content})
        usage = result.usage
        record.update({
            "answer": result.output,
            "elapsed_seconds": elapsed_seconds,
            "usage": {"input_tokens": usage.input_tokens,
                      "output_tokens": usage.output_tokens,
                      "requests": usage.requests,
                      "generation_cost_usd": usage.cost},
            "events": events,
        })
        original_trace(result, elapsed_seconds)

    capture = io.StringIO()
    original_argv = sys.argv
    agent.print_trace = save_trace
    sys.argv = ["app.agent", *cli_args]
    exit_code = 0
    try:
        with redirect_stdout(capture), redirect_stderr(capture):
            try:
                agent.main()
            except SystemExit as exc:
                exit_code = exc.code if isinstance(exc.code, int) else 1
                if exc.code and not isinstance(exc.code, int):
                    print(str(exc))
            except Exception as exc:
                exit_code = 1
                print(f"{type(exc).__name__}: {exc}")
    finally:
        agent.print_trace = original_trace
        sys.argv = original_argv

    record["exit_code"] = exit_code
    log_text = capture.getvalue()
    json_text = json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n"
    key = os.getenv("GEMINI_API_KEY")
    if key:
        log_text = log_text.replace(key, "[REDACTED]")
        json_text = json_text.replace(key, "[REDACTED]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".log").write_text(log_text, encoding="utf-8")
    args.output.with_suffix(".json").write_text(json_text, encoding="utf-8")
    print(f"{args.output}: exit={exit_code}")
    print(log_text)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
