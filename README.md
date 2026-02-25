# Finetune Qwen2.5 for Structured API Invocation

We investigate finetuning of Qwen2.5-7B-Instruct using LoRA on Gorilla APIBench dataset. The objective is to improve **structured** API-style generation and semantic **alignment**. Evaluation compares baseline and LoRA-adapted models using **lexical**, **structural**, and **functional** metrics.

### Background

Gorilla enables LLMs to invoke APIs by generating (semantically and syntactically) correct API calls from natural language queries.  

**Gorilla API Store** and **APIBench** dataset aim to enhance LLM tool-use capabilities through curated API examples across 1,600+ APIs.

- _Gorilla enables LLMs to use tools by invoking APIs and reduces hallucination in API generation._

### Objective

Improve:

- ***Structured API response formatting*** (e.g., `<<<domain>>>`, `<<<api_call>>>`, `<<<code>>>`)
- ***Semantic similarity*** to reference API solutions
- ***Functional correctness*** in API usage patterns

### Methodology

The Model :
- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Fine-tuning method: **LoRA (PEFT)**
- Trainable parameters: ~0.06% of total model parameters
- LoRA adapters applied to: `q_proj, k_proj, v_proj, o_proj`

Training Setup :
- Max sequence length: 1024 tokens
- Masked loss: train only on assistant outputs
- FP16 mixed precision
- Single epoch fine-tuning

Evaluation :
- Baseline and LoRA models are evaluated on APIBench samples.
- Metrics:
    - **Jaccard similarity** (lexical overlap)
    - **ROUGE-L F1** (sequence-aware similarity)
    - **Tag compliance score** (structured formatting adherence)
    - **Functional API score** (presence of valid API call patterns)

### Results 
_(on 500 sample evaluation)_ LoRA fine-tuning significantly improved structural compliance and semantic similarity without increasing output length.

| Metric | Baseline | LoRA |
|---------|----------|------|
| Jaccard | 0.157 | 0.230 |
| ROUGE-L F1 | 0.135 | 0.195 |
| Tag compliance (0–5) | 0.00 | 2.91 |


- Key Observations
    - Baseline model ignores (APIBench) structural format.
    - LoRA adapters successfully internalize format constraints.
    - Semantic alignment improves consistently across evaluation samples.
    - Functional API invocation patterns increase after fine-tuning.

_Parameter-efficient fine-tuning with LoRA effectively adapts Qwen2.5 toward structured API invocation tasks as defined in Gorilla APIBench._

Future work includes:
- Multi-epoch training
- Larger evaluation sets
- Exact API-call correctness validation
- Functional execution-based evaluation