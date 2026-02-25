import json
import time
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data import load_jsonl, parse_apibench_code

CONFIG_PATH = Path("configs/run.json")
SYSTEM = "You are a helpful API writer who can write APIs based on requirements."

def load_eval_examples(files: List[str]):
    rows = []
    for fp in files:
        rows.extend(load_jsonl(fp))
    examples = []
    for r in rows:
        inst, out = parse_apibench_code(r.get("code", ""))
        if inst and out:
            examples.append({"instruction": inst, "reference": out})
    return examples

def build_input(tokenizer, instruction: str):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )

@torch.no_grad()
def generate(model, tokenizer, input_ids, max_new_tokens=120):
    input_ids = input_ids.to(model.device)
    gen = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(gen[0], skip_special_tokens=False)

def fmt_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    m = seconds // 60
    s = seconds % 60
    return f"{m:02d}:{s:02d}"

def main():
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    eval_examples = load_eval_examples(cfg["eval_files"])

    MAX_EVAL = min(500, len(eval_examples))
    print(f"Baseline-only eval on {MAX_EVAL} examples")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading baseline model...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        device_map="auto",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    output_path = Path(cfg["output_dir"]) / "eval_generation_baseline_only.jsonl"
    print("Saving to:", output_path)

    total = 0.0
    t0 = time.perf_counter()
    with output_path.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(eval_examples[:MAX_EVAL], start=1):
            t = time.perf_counter()
            ids = build_input(tokenizer, ex["instruction"])
            out = generate(model, tokenizer, ids, max_new_tokens=120)
            dt = time.perf_counter() - t
            total += dt
            avg = total / i
            eta = (MAX_EVAL - i) * avg
            print(f"[{i}/{MAX_EVAL}] sample={fmt_hms(dt)} avg={fmt_hms(avg)} ETA={fmt_hms(eta)}")

            f.write(json.dumps({
                "instruction": ex["instruction"],
                "reference": ex["reference"],
                "baseline_output": out,
            }, ensure_ascii=False) + "\n")

    print("Done. Total:", fmt_hms(time.perf_counter() - t0))

if __name__ == "__main__":
    main()