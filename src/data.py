import json, re
from datasets import Dataset

SYSTEM = "You are a helpful API writer who writes correct API calls and code snippets."

def parse_apibench_code(code: str):
    m_inst = re.search(r"###\s*Instruction\s*:\s*(.*?)\n###\s*Output\s*:", code, flags=re.DOTALL | re.IGNORECASE)
    instruction = m_inst.group(1).strip() if m_inst else None

    m_out = re.search(r"###\s*Output\s*:\s*(.*)$", code, flags=re.DOTALL | re.IGNORECASE)
    output = m_out.group(1).strip() if m_out else None

    return instruction, output

def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def to_messages(instruction: str, output: str):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]

def build_raw_dataset(files: list[str]) -> Dataset:
    data = []
    for fp in files:
        data.extend(load_jsonl(fp))
    data_min = [{"code": d.get("code", "")} for d in data if isinstance(d, dict) and "code" in d]
    return Dataset.from_list(data_min)