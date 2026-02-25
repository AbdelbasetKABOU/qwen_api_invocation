import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER = "./outputs/qwen2p5_apibench_lora/final_adapter"

def main():
    base = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
    tok.pad_token = tok.eos_token
    base.config.pad_token_id = tok.pad_token_id

    model = PeftModel.from_pretrained(base, ADAPTER)
    model.eval()

    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))

    # This is the key confirmation:
    print("\nDevice map:")
    print(model.base_model.model.hf_device_map)

    # Memory snapshot (rough usage)
    print("\nGPU memory allocated (GB):")
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"GPU{i}: allocated={alloc:.2f} reserved={reserved:.2f}")

if __name__ == "__main__":
    main()