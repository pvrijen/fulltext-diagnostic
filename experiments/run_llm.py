#!/usr/bin/env python3
"""Few-shot LLM screening with multiple SOTA models.

Usage (one model per GPU):
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python papers/fewshot_llm_multi.py --model llama70b
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python papers/fewshot_llm_multi.py --model qwen72b
  CUDA_VISIBLE_DEVICES=2 .venv/bin/python papers/fewshot_llm_multi.py --model gemma27b
"""
import json, os, sys, random, time, argparse
import numpy as np, torch
from pathlib import Path
from sklearn.metrics import average_precision_score

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from run_rq2_unified import load_data, make_cv_splits, set_seeds

MODELS = {
    "llama70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen72b": "Qwen/Qwen2.5-72B-Instruct",
    "gemma27b": "google/gemma-2-27b-it",
    "phi14b": "microsoft/Phi-3-medium-128k-instruct",
    "yi34b": "01-ai/Yi-1.5-34B-Chat",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
}
DATASETS = ["pfas", "bpa", "cd012661", "cd012069", "cd008122", "cd011431"]
SEED = 42
N_SHOT = 5

TOPICS = {
    "pfas": "Immunotoxicity of perfluoroalkyl substances (PFOA/PFOS)",
    "bpa": "Metabolic effects of bisphenol A",
    "cd012661": "Screening for gestational diabetes mellitus",
    "cd012069": "Methylphenidate adverse events in children and adolescents with ADHD",
    "cd008122": "Rapid diagnostic tests for detecting Plasmodium falciparum malaria",
    "cd011431": "Rapid diagnostic tests for detecting Plasmodium vivax malaria",
}

FEW_SHOT_TEMPLATE = """You are screening articles for a systematic review on: {topic}
For each study, answer 'yes' (include) or 'no' (exclude).

{examples}
Study: {text}
The answer is"""

EXAMPLE_TEMPLATE = """Study: {text}
The answer is {label}"""

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset list")
args = parser.parse_args()
if args.datasets:
    DATASETS = args.datasets.split(",")

model_id = MODELS[args.model]
OUT = Path(f"papers/output_fewshot_llm_{args.model}")
OUT.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{args.model}|{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def build_examples(texts, labels, n_per_class, rng):
    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    neg_idx = [i for i, l in enumerate(labels) if l == 0]
    rng.shuffle(pos_idx); rng.shuffle(neg_idx)
    selected = pos_idx[:n_per_class] + neg_idx[:n_per_class]
    rng.shuffle(selected)
    examples = []
    for i in selected:
        words = texts[i].split()[:200]
        label = "yes" if labels[i] == 1 else "no"
        examples.append(EXAMPLE_TEMPLATE.format(text=' '.join(words), label=label))
    return '\n\n'.join(examples)

log(f"Loading {model_id} (4-bit)...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, quantization_config=quant_config,
    device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
model.eval()

yes_ids, no_ids = [], []
for w in ["yes", "Yes", "YES", " yes", " Yes"]:
    yes_ids.extend(tokenizer.encode(w, add_special_tokens=False))
for w in ["no", "No", "NO", " no", " No"]:
    no_ids.extend(tokenizer.encode(w, add_special_tokens=False))
yes_ids = list(set(yes_ids)); no_ids = list(set(no_ids))
log(f"Loaded. Yes IDs: {len(yes_ids)}, No IDs: {len(no_ids)}")

def score_one(prompt, max_tokens=8192):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=max_tokens).to(model.device)
    try:
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1, :]
            y = max(logits[t].item() for t in yes_ids)
            n = max(logits[t].item() for t in no_ids)
            probs = torch.softmax(torch.tensor([y, n]), dim=0)
            return probs[0].item()
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

rows = []
for ds in DATASETS:
    set_seeds(SEED)
    df, ft, med, labels = load_data(ds)
    splits = make_cv_splits(labels, SEED)

    for fold in range(5):
        tag = f"{ds}_f{fold}"
        result_path = OUT / f"{tag}.json"
        if result_path.exists():
            rows.append(json.load(open(result_path)))
            log(f"SKIP {tag}")
            continue

        train_idx, test_idx = splits[fold]
        y_test = [labels[i] for i in test_idx]
        rng = random.Random(SEED + fold)
        train_texts = [med[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        example_str = build_examples(train_texts, train_labels, N_SHOT, rng)

        log(f"{tag} ({len(test_idx)} test docs)")
        scores = []
        for j, idx in enumerate(test_idx):
            words = med[idx].split()[:400]
            prompt = FEW_SHOT_TEMPLATE.format(
                topic=TOPICS[ds], examples=example_str, text=' '.join(words))
            s = score_one(prompt)
            scores.append(s if s is not None else 0.5)
            if (j + 1) % 200 == 0:
                log(f"  {j+1}/{len(test_idx)}")

        prauc = average_precision_score(y_test, scores)
        log(f"  PR-AUC: {prauc:.3f}")
        r = {"dataset": ds, "fold": fold, "model": args.model, "few_shot": prauc,
             "n_test": len(test_idx), "n_pos": sum(y_test)}
        rows.append(r)
        result_path.write_text(json.dumps(r, indent=2))

# Aggregate
log(f"\nAGGREGATE ({args.model}):")
fs = np.mean([r['few_shot'] for r in rows])
log(f"  Few-shot PR-AUC: {fs:.3f} (BioMedBERT: 0.703)")
for ds in DATASETS:
    dr = [r for r in rows if r['dataset'] == ds]
    log(f"  {ds:10s}: {np.mean([r['few_shot'] for r in dr]):.3f}")

with open(OUT / "summary.json", "w") as f:
    json.dump(rows, f, indent=2)
