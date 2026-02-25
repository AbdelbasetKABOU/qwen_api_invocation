import json
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq  # padding only

from peft import LoraConfig, get_peft_model, TaskType

def main():
    cfg = json.loads(Path("configs/run.json").read_text(encoding="utf-8"))

    ds = load_from_disk(cfg["tokenized_dir"])

    tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
    tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        device_map="auto",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    base.config.use_cache = False
    base.config.pad_token_id = tok.pad_token_id

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        target_modules=cfg["target_modules"],
    )

    base.enable_input_require_grads()
    model = get_peft_model(base, lora_config)

    # Fix "Attempting to unscale FP16 gradients": keep trainable params in fp32
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    model.print_trainable_parameters()

    # Padding-only collator (keeps your labels)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tok, padding=True)

    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["num_train_epochs"],
        fp16=True,
        logging_steps=25,
        save_steps=200,
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=True,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
        optim="adamw_hf",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        tokenizer=tok,
        data_collator=data_collator,
    )

    trainer.train()

    out_dir = str(Path(cfg["output_dir"]) / "final_adapter")
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print("Saved adapter to:", out_dir)

if __name__ == "__main__":
    main()