#!/usr/bin/env python3
"""Zero-shot and few-shot LLM screening baselines (abstract-only input).

Scores each test abstract by the next-token probability of "yes" vs "no"
after a screening prompt. Models are loaded in 4-bit (bitsandbytes).

Usage (from the repository root, one model per GPU):
  python experiments/run_llm.py --model qwen2.5-32b --mode zeroshot,fewshot
  python experiments/run_llm.py --model mistral-7b --mode fewshot
"""
import argparse
import json
import random
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from _helpers import DATASET_NAMES, OUT_ROOT, SEED, load_data, make_cv_splits, set_seeds

MODELS = {
    "qwen2.5-32b": "Qwen/Qwen2.5-32B-Instruct",
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    # Additional models evaluated during development
    "llama70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen72b": "Qwen/Qwen2.5-72B-Instruct",
    "gemma27b": "google/gemma-2-27b-it",
    "phi14b": "microsoft/Phi-3-medium-128k-instruct",
    "yi34b": "01-ai/Yi-1.5-34B-Chat",
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
}
N_SHOT = 5

TOPICS = {
    "pfas": "Immunotoxicity of perfluoroalkyl substances (PFOA/PFOS)",
    "bpa": "Metabolic effects of bisphenol A",
    "cd012661": "Screening for gestational diabetes mellitus",
    "cd012069": "Methylphenidate adverse events in children and adolescents with ADHD",
    "cd008122": "Rapid diagnostic tests for detecting Plasmodium falciparum malaria",
    "cd011431": "Rapid diagnostic tests for detecting Plasmodium vivax malaria",
}

ZERO_SHOT_TEMPLATE = """You are screening articles for a systematic review on: {topic}
For each study, answer 'yes' (include) or 'no' (exclude).

Study: {text}
The answer is"""

FEW_SHOT_TEMPLATE = """You are screening articles for a systematic review on: {topic}
For each study, answer 'yes' (include) or 'no' (exclude).

{examples}
Study: {text}
The answer is"""

EXAMPLE_TEMPLATE = """Study: {text}
The answer is {label}"""


def build_examples(texts, labels, n_per_class, rng):
    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    neg_idx = [i for i, l in enumerate(labels) if l == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    selected = pos_idx[:n_per_class] + neg_idx[:n_per_class]
    rng.shuffle(selected)
    examples = []
    for i in selected:
        words = texts[i].split()[:200]
        label = "yes" if labels[i] == 1 else "no"
        examples.append(EXAMPLE_TEMPLATE.format(text=" ".join(words), label=label))
    return "\n\n".join(examples)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(MODELS.keys()))
    parser.add_argument("--mode", default="zeroshot,fewshot",
                        help="Comma-separated subset of: zeroshot, fewshot")
    parser.add_argument("--datasets", default=",".join(DATASET_NAMES),
                        help="Comma-separated dataset list")
    args = parser.parse_args()
    modes = [m.strip() for m in args.mode.split(",")]
    for m in modes:
        if m not in ("zeroshot", "fewshot"):
            parser.error(f"unknown mode {m!r}")
    datasets = args.datasets.split(",")

    model_id = MODELS[args.model]
    out = OUT_ROOT / "llm" / args.model
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{args.model}|{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"Loading {model_id} (4-bit)...")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant_config,
        device_map="auto", trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.eval()

    yes_ids, no_ids = set(), set()
    for w in ["yes", "Yes", "YES", " yes", " Yes"]:
        yes_ids.update(tokenizer.encode(w, add_special_tokens=False))
    for w in ["no", "No", "NO", " no", " No"]:
        no_ids.update(tokenizer.encode(w, add_special_tokens=False))
    yes_ids, no_ids = sorted(yes_ids), sorted(no_ids)
    log(f"Loaded. Yes IDs: {len(yes_ids)}, No IDs: {len(no_ids)}")

    def score_one(prompt, max_tokens=8192):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=max_tokens).to(model.device)
        try:
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1, :]
                y = max(logits[t].item() for t in yes_ids)
                n = max(logits[t].item() for t in no_ids)
                return torch.softmax(torch.tensor([y, n]), dim=0)[0].item()
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return None

    rows = []
    for ds in datasets:
        set_seeds(SEED)
        df, ft, med, labels = load_data(ds)
        splits = make_cv_splits(labels, SEED)

        for fold in range(5):
            tag = f"{ds}_f{fold}"
            result_path = out / f"{tag}.json"
            r = json.load(open(result_path)) if result_path.exists() else {
                "dataset": ds, "fold": fold, "model": args.model}
            train_idx, test_idx = splits[fold]
            y_test = [labels[i] for i in test_idx]
            r["n_test"], r["n_pos"] = len(test_idx), int(sum(y_test))

            for mode in modes:
                key = "zero_shot" if mode == "zeroshot" else "few_shot"
                if key in r:
                    log(f"SKIP {tag} {mode}")
                    continue
                if mode == "fewshot":
                    rng = random.Random(SEED + fold)
                    example_str = build_examples(
                        [med[i] for i in train_idx],
                        [labels[i] for i in train_idx], N_SHOT, rng)
                log(f"{tag} {mode} ({len(test_idx)} test docs)")
                scores = []
                for j, idx in enumerate(test_idx):
                    text = " ".join(med[idx].split()[:400])
                    if mode == "fewshot":
                        prompt = FEW_SHOT_TEMPLATE.format(
                            topic=TOPICS[ds], examples=example_str, text=text)
                    else:
                        prompt = ZERO_SHOT_TEMPLATE.format(topic=TOPICS[ds], text=text)
                    s = score_one(prompt)
                    scores.append(s if s is not None else 0.5)
                    if (j + 1) % 200 == 0:
                        log(f"  {j + 1}/{len(test_idx)}")
                r[key] = float(average_precision_score(y_test, scores))
                log(f"  {mode} PR-AUC: {r[key]:.3f}")
                result_path.write_text(json.dumps(r, indent=2))
            rows.append(r)

    log(f"\nAGGREGATE ({args.model}):")
    for key in ["zero_shot", "few_shot"]:
        vals = [r[key] for r in rows if key in r]
        if vals:
            log(f"  {key}: PR-AUC {np.mean(vals):.3f} (n={len(vals)})")
            for ds in datasets:
                dv = [r[key] for r in rows if r["dataset"] == ds and key in r]
                if dv:
                    log(f"    {ds:10s}: {np.mean(dv):.3f}")

    (out / "summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
