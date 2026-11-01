"""Run lecture 2's notebook locally, keeping its data and scoring rules.

Example (use a separate environment with the notebook's pinned dependencies):
    .venv/lecture02/bin/python scripts/run_lora_local.py --name baseline
    .venv/lecture02/bin/python scripts/run_lora_local.py --name lr_1e4 --lr 0.0001
    .venv/lecture02/bin/python scripts/run_lora_local.py --name cpu_train_mps_eval --device cpu --eval-device mps

Each command starts a fresh process and loads the original base-model revision.
Large model/checkpoint files stay under the ignored .venv directory.
"""

import argparse
import ast
import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import accelerate
import datasets
import peft
import torch
import transformers
import trl

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks/lora_format_tuning.ipynb"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, choices=["baseline", "lr_1e4", "cpu_baseline", "cpu_train_mps_eval"])
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--eval-device", choices=["mps", "cpu"], help="Defaults to the training device")
    args = parser.parse_args()
    eval_device = args.eval_device or args.device
    if args.device == "cpu":
        torch.set_num_threads(4)
    work = ROOT / ".venv" / "lecture02-runs" / args.name
    if work.exists():
        raise SystemExit(f"Output directory already exists; preserving previous run: {work}")
    work.mkdir(parents=True)
    notebook_bytes = NOTEBOOK.read_bytes()
    notebook = json.loads(notebook_bytes)
    namespace = {
        "__name__": "__main__",
        "transformers": transformers, "peft": peft, "trl": trl, "torch": torch,
        "display": lambda value: print(value.to_string(), flush=True),
    }
    run_info = {
        "name": args.name, "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "notebook_sha256": hashlib.sha256(notebook_bytes).hexdigest(),
        "versions": {m.__name__: m.__version__ for m in (torch, transformers, peft, trl, accelerate, datasets)},
        "local_changes": ["local output directory", "explicit fp16=False and bf16=False", "per-step numeric checks and logging"],
        "requested_device": args.device,
        "eval_device": eval_device,
        "torch_num_threads": torch.get_num_threads(),
        "status": "started",
    }

    def write_json(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def save_info():
        write_json(work / "run_info.json", run_info)

    def source(marker):
        matches = ["".join(cell["source"]) for cell in notebook["cells"]
                   if cell["cell_type"] == "code" and marker in "".join(cell["source"])]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one notebook cell for {marker!r}; found {len(matches)}")
        return matches[0]

    def execute(text):
        exec(compile(text, str(NOTEBOOK), "exec"), namespace)

    class LocalOutput(ast.NodeTransformer):
        def visit_Dict(self, node):
            self.generic_visit(node)
            for index, key in enumerate(node.keys):
                if isinstance(key, ast.Constant) and key.value == "lr":
                    node.values[index] = ast.copy_location(ast.Constant(args.lr), node.values[index])
            return node

        def visit_Constant(self, node):
            if node.value == "/content/qlora_lab":
                return ast.copy_location(ast.Constant(str(work)), node)
            return node

    class NumericChecks(transformers.TrainerCallback):
        def on_pre_optimizer_step(self, training_args, state, control, **kwargs):
            for name, parameter in kwargs["model"].named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all().item():
                        raise RuntimeError(f"Non-finite gradient at step {state.global_step + 1}: {name}")

        def on_log(self, training_args, state, control, logs=None, **kwargs):
            if logs is None:
                return
            for key, value in logs.items():
                if isinstance(value, (int, float)) and not math.isfinite(value):
                    raise RuntimeError(f"Non-finite {key} at step {state.global_step}: {value}")
            run_info["latest_training_log"] = {"step": state.global_step, **logs}
            save_info()
            print("TRAIN_STEP", json.dumps(run_info["latest_training_log"], ensure_ascii=False, allow_nan=False), flush=True)

    save_info()
    try:
        if "mps" in (args.device, eval_device) and not torch.backends.mps.is_available():
            raise RuntimeError("Apple GPU (MPS) is not available; refusing an unnoticed CPU run")
        config_tree = LocalOutput().visit(ast.parse(source("CONFIG = {")))
        exec(compile(ast.fix_missing_locations(config_tree), str(NOTEBOOK), "exec"), namespace)
        namespace["CONFIG"]["lr"] = args.lr
        execute(source("FAQ = {"))
        assert (len(namespace["train"]), len(namespace["valid"]), len(namespace["test"])) == (14, 7, 21)
        transformers.set_seed(namespace["CONFIG"]["seed"])
        model_source = source('DEVICE = "cuda"')
        original_device = 'DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")'
        assert model_source.count(original_device) == 1
        execute(model_source.replace(original_device, f"DEVICE = {eval_device!r}"))
        model = namespace["model"]
        model.set_attn_implementation("sdpa")
        assert model.device.type == eval_device
        print("PHASE before_evaluation", flush=True)
        run_info["status"] = "before_evaluation"
        save_info()
        execute(source("def check_format("))
        run_info["before"] = namespace["before_summary"]
        model.to(args.device)
        if args.device == "cpu" and eval_device == "mps":
            torch.mps.empty_cache()
        run_info["status"] = "training"
        save_info()
        training = source("from peft import LoraConfig")
        assert training.count("fp16=USE_AMP,") == 1
        training = training.replace("fp16=USE_AMP,", f"fp16=USE_AMP, bf16=False, use_cpu={args.device == 'cpu'},")
        setup, finish = training.split("t0 = time.time()", 1)
        execute(setup)
        trainer = namespace["trainer"]
        assert trainer.args.device.type == args.device
        assert not trainer.args.fp16 and not trainer.args.bf16
        trainer.args.logging_nan_inf_filter = False
        trainer.args.logging_steps = 1
        trainer.args.disable_tqdm = True
        trainer.add_callback(NumericChecks())
        run_info["actual_training_args"] = {
            "device": str(trainer.args.device), "fp16": trainer.args.fp16,
            "bf16": trainer.args.bf16, "mixed_precision": trainer.accelerator.mixed_precision,
            "batch_size": trainer.args.per_device_train_batch_size,
            "gradient_accumulation_steps": trainer.args.gradient_accumulation_steps,
            "learning_rate": trainer.args.learning_rate, "epochs": trainer.args.num_train_epochs,
        }
        save_info()
        print("PHASE training", json.dumps(run_info["actual_training_args"]), flush=True)
        t0 = time.monotonic()
        execute("t0 = time.time()" + finish)
        if args.device == "mps":
            torch.mps.synchronize()
        training_seconds = time.monotonic() - t0
        assert trainer.state.global_step == 40
        assert all(torch.isfinite(p).all().item() for p in model.parameters() if p.requires_grad)
        history = trainer.state.log_history
        training_losses = [row["loss"] for row in history if "loss" in row]
        assert training_losses and all(math.isfinite(value) for value in training_losses)
        training_summary = {
            "global_steps": trainer.state.global_step,
            "epochs": trainer.state.epoch,
            "seconds": next(row["train_runtime"] for row in reversed(history) if "train_runtime" in row),
            "train_and_save_seconds": round(training_seconds, 3),
            "final_logged_loss": training_losses[-1],
            "average_training_loss": next(row["train_loss"] for row in reversed(history) if "train_loss" in row),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
        }
        write_json(work / "training_log.json", history)
        run_info["training"] = training_summary
        run_info["status"] = "after_evaluation"
        save_info()
        print("PHASE after_evaluation", flush=True)
        namespace["tuned"].to(eval_device)
        execute(source('after_summary, after_rows = evaluate(tuned, "after")'))
        print("PHASE knowledge_questions", flush=True)
        execute(source("KNOWLEDGE_QUESTIONS = ["))
        # The notebook tests FAQ membership against test-only gold. The control question is
        # in train instead, so use the actual FAQ templates for this descriptive flag.
        known_questions = {q for templates, _ in namespace["GROUNDED"].values() for q in templates}
        for row in namespace["knowledge_rows"]:
            row["FAQ にある"] = row["質問"] in known_questions
        execute(source("result = {"))
        result = namespace["result"]
        result["training"] = training_summary
        result["execution"] = {
            "device": args.device, "platform": platform.platform(),
            "eval_device": eval_device,
            "torch_num_threads": torch.get_num_threads(),
            "fp16": False, "bf16": False, "attention": "sdpa", "assisted_by": "Codex",
            "notebook_sha256": run_info["notebook_sha256"],
            "knowledge_membership_fix": "FAQ control question is in train, not test-only gold",
        }
        write_json(work / "result.json", result)
        run_info["status"] = "completed"
        run_info["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_info()
        print("RUN_COMPLETED", str(work / "result.json"), flush=True)
    except Exception as exc:
        run_info["status"] = "failed"
        run_info["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save_info()
        raise


if __name__ == "__main__":
    main()
