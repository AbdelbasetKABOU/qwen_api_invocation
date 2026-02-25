import json
import re
from pathlib import Path
from statistics import mean

from rouge_score import rouge_scorer

BASELINE_PATH = Path("./outputs/qwen2p5_apibench_lora/eval_generation_baseline_only.jsonl")
LORA_PATH     = Path("./outputs/qwen2p5_apibench_lora/eval_generation_lora_only.jsonl")

TAGS = ["<<<domain>>>", "<<<api_call>>>", "<<<api_provider>>>", "<<<explanation>>>", "<<<code>>>"]
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

def load_jsonl(p: Path):
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())

def words(s: str):
    s = s.lower()
    s = re.sub(r"[^a-z0-9_<>]+", " ", s)
    toks = [t for t in s.split() if t]
    return set(toks)

def jaccard(a: str, b: str) -> float:
    A, B = words(a), words(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)

def rougeL_f1(pred: str, ref: str) -> float:
    # rouge_score returns precision/recall/fmeasure
    return scorer.score(ref, pred)["rougeL"].fmeasure

def tag_hits(text: str) -> int:
    return sum(t in text for t in TAGS)

def main():
    base_rows = load_jsonl(BASELINE_PATH)
    lora_rows = load_jsonl(LORA_PATH)

    base_by_inst = {normalize_text(r["instruction"]): r for r in base_rows}
    lora_by_inst = {normalize_text(r["instruction"]): r for r in lora_rows}

    common = sorted(set(base_by_inst.keys()) & set(lora_by_inst.keys()))
    if not common:
        print("No common instructions found between baseline and LoRA files.")
        return

    results = []
    for inst in common:
        b = base_by_inst[inst]
        l = lora_by_inst[inst]

        ref = b.get("reference", "")
        b_out = b.get("baseline_output", "")
        l_out = l.get("lora_output", "")

        results.append({
            "b_jacc": jaccard(b_out, ref),
            "l_jacc": jaccard(l_out, ref),
            "b_rougeL": rougeL_f1(b_out, ref),
            "l_rougeL": rougeL_f1(l_out, ref),
            "b_tags": tag_hits(b_out),
            "l_tags": tag_hits(l_out),
            "b_len": len(b_out),
            "l_len": len(l_out),
        })

    print(f"Compared {len(results)} examples\n")
    print("Averages:")
    print(f"- Jaccard vs reference: baseline={mean(r['b_jacc'] for r in results):.3f} | lora={mean(r['l_jacc'] for r in results):.3f}")
    print(f"- ROUGE-L F1 vs reference: baseline={mean(r['b_rougeL'] for r in results):.3f} | lora={mean(r['l_rougeL'] for r in results):.3f}")
    print(f"- Tag hits (out of {len(TAGS)}): baseline={mean(r['b_tags'] for r in results):.2f} | lora={mean(r['l_tags'] for r in results):.2f}")
    print(f"- Output length (chars): baseline={mean(r['b_len'] for r in results):.0f} | lora={mean(r['l_len'] for r in results):.0f}")

if __name__ == "__main__":
    main()