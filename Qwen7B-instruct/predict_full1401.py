#!/usr/bin/env python3
"""
Run inference-only prediction on all 1401 patients (input_1600, corrected -
see the input_1600 truncation fix where 494 notes were replaced with the
complete versions from input_500) using the freshly fine-tuned
46-CDE / 1401-patient model (no retraining - see
Qwen7B-instruct/model/qwen_full_7B_cde1600raw_20260722_143842_local/training_config.json).

pred_cde_1600raw/ only has the 280 held-out eval patients; this script
regenerates predictions for all 1401 patients from scratch.

Output: Qwen7B-instruct/pred_cde_1600raw_full1401/*_pred.txt (1401 files)
"""
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(__file__).resolve().parent / "model" / "qwen_full_7B_cde1600raw_20260722_143842_local" / "final_model"
DATA_INPUT = REPO_ROOT / "data_training" / "500_1600_raw_harmo" / "input_1600"
OUT_DIR = Path(__file__).resolve().parent / "pred_cde_1600raw_full1401"

INSTRUCTION = "Using the following note, extract structured key-value pairs about the patient's symptoms and diagnoses:"


def get_id(name: str):
    m = re.match(r"(B\d+)", name)
    return m.group(1) if m else None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = sorted(DATA_INPUT.glob("*.txt"))
    print(f"Found {len(input_files)} input notes in {DATA_INPUT}")

    to_generate = []
    skipped = 0
    for inp_file in input_files:
        cid = get_id(inp_file.name)
        if not cid:
            continue
        if (OUT_DIR / f"{cid}_pred.txt").exists():
            skipped += 1
            continue
        to_generate.append((cid, inp_file))
    print(f"Skipping {skipped} already-generated predictions (resume after interruption)")
    print(f"Will generate {len(to_generate)} predictions")

    if not to_generate:
        print("Nothing left to generate, done.")
        return

    print(f"Loading tokenizer/model from {MODEL_PATH} ...")
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tok,
        device=0 if torch.cuda.is_available() else -1,
        return_full_text=False,
        pad_token_id=tok.pad_token_id,
        dtype=torch.bfloat16,
    )

    for i, (cid, inp_file) in enumerate(to_generate, 1):
        note = inp_file.read_text(encoding="utf-8").strip()
        prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": note},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        out = gen(prompt, max_new_tokens=500, do_sample=False)
        txt = out[0]["generated_text"].strip()
        (OUT_DIR / f"{cid}_pred.txt").write_text(txt, encoding="utf-8")
        if i % 20 == 0 or i == len(to_generate):
            print(f"Generated {i}/{len(to_generate)} -> {cid}")

    print(f"\nDone. {OUT_DIR} now has {len(list(OUT_DIR.glob('*_pred.txt')))} predictions.")


if __name__ == "__main__":
    main()
