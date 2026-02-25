import json
import time
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.data import load_jsonl, parse_apibench_code

CONFIG_PATH = Path("configs/run.json")
SYSTEM = "You are a helpful API writer who can write APIs based on requirements."

# -----------------------------
# Utilities
# -----------------------------

def load_eval_examples(files: List[str]):
    print("Loading eval files...")
    rows = []
    for fp in files:
        print(f"  - {fp}")
        rows.extend(load_jsonl(fp))

    examples = []
    for r in rows:
        inst, out = parse_apibench_code(r.get("code", ""))
        if inst and out:
            examples.append({"instruction": inst, "reference": out})

    print(f"Loaded {len(examples)} valid eval examples")
    return examples


def build_input(tokenizer, instruction: str):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )


@torch.no_grad()
def generate(model, tokenizer, input_ids, max_new_tokens: int = 120):
    input_ids = input_ids.to(model.device)
    gen = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # deterministic
    )
    return tokenizer.decode(gen[0], skip_special_tokens=False)


def fmt_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# -----------------------------
# Main
# -----------------------------
def main():
    print("=" * 60)
    print("STEP A (LoRA-only): Generate on eval dataset")
    print("=" * 60)

    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    eval_examples = load_eval_examples(cfg["eval_files"])

    MAX_EVAL = min(500, len(eval_examples))
    print(f"Evaluating on {MAX_EVAL} examples\n")

    # Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base + adapter (single model)
    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        device_map="auto",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    base.config.pad_token_id = tokenizer.pad_token_id
    base.config.use_cache = True  # generation speed

    print("Loading LoRA adapter...")
    adapter_dir = str(Path(cfg["output_dir"]) / "final_adapter")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    # Quick sanity: show if any CPU offload happened
    if hasattr(model.base_model.model, "hf_device_map"):
        dm = model.base_model.model.hf_device_map
        cpu_offload = any(v == "cpu" for v in dm.values())
        print(f"Device map loaded. CPU offload: {cpu_offload}")

    print("Model ready.\n")

    output_path = Path(cfg["output_dir"]) / "eval_generation_lora_only.jsonl"
    print(f"Saving results to: {output_path}\n")

    t0 = time.perf_counter()
    total_gen_time = 0.0

    with output_path.open("w", encoding="utf-8") as f:
        for idx, ex in enumerate(eval_examples[:MAX_EVAL], start=1):
            t_ex = time.perf_counter()

            input_ids = build_input(tokenizer, ex["instruction"])
            lora_out = generate(model, tokenizer, input_ids, max_new_tokens=120)

            ex_time = time.perf_counter() - t_ex
            total_gen_time += ex_time

            avg = total_gen_time / idx
            remaining = MAX_EVAL - idx
            eta = remaining * avg

            print(f"[{idx:>2}/{MAX_EVAL}] sample={fmt_hms(ex_time)} avg={fmt_hms(avg)} ETA={fmt_hms(eta)}")

            record = {
                "instruction": ex["instruction"],
                "reference": ex["reference"],
                "lora_output": lora_out,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    elapsed = time.perf_counter() - t0
    print("\nEvaluation finished.")
    print(f"Total elapsed: {fmt_hms(elapsed)} | Avg/sample: {fmt_hms(total_gen_time / MAX_EVAL)}")
    print("Results saved successfully.")


if __name__ == "__main__":
    main()