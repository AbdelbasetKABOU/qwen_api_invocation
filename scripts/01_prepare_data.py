import json
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer
from src.data import build_raw_dataset, parse_apibench_code, to_messages

def preprocess_row(ex, tokenizer, max_length: int):
    inst, out = parse_apibench_code(ex["code"])
    if not inst or not out:
        return {"input_ids": [], "labels": [], "attention_mask": []}

    messages = to_messages(inst, out)

    prompt_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    full_ids   = tokenizer.apply_chat_template(messages,    tokenize=True, add_generation_prompt=False)

    prompt_ids = prompt_ids[:max_length]
    full_ids   = full_ids[:max_length]

    labels = full_ids.copy()
    mask_len = min(len(prompt_ids), len(labels))
    labels[:mask_len] = [-100] * mask_len

    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}

def main():
    cfg = json.loads(Path("configs/run.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    tok.pad_token = tok.eos_token

    raw_ds = build_raw_dataset(cfg["raw_files"])

    tokenized = raw_ds.map(lambda ex: preprocess_row(ex, tok, cfg["max_length"]))
    tokenized = tokenized.filter(lambda x: len(x["input_ids"]) > 0)
    tokenized = tokenized.remove_columns(["code"])

    out_dir = cfg["tokenized_dir"]
    tokenized.save_to_disk(out_dir)
    print("Saved tokenized dataset to:", out_dir)
    print(tokenized)

if __name__ == "__main__":
    main()