#!/usr/bin/env python3
# llama-generate_all.py
# Generate predictions using fine-tuned Llama model

from pathlib import Path
import os, re, json, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ==================== PARAMS ====================
MODEL_SHORT = os.getenv("MODEL_SHORT", "llama_full_8B")
INSTRUCTION = os.getenv("INSTRUCTION", "Using the following note, extract structured key-value pairs about the patient's symptoms and diagnoses:")
MAX_PRED    = int(os.getenv("MAX_PRED", "0"))  # 0 = all

# Determine paths
BASE_DIR     = Path(__file__).parent.parent
TRAINING_DIR = BASE_DIR / "data_training"
DATA_INPUT   = TRAINING_DIR / "data_input"
DATA_OUTPUT  = TRAINING_DIR / "data_output_clean"

def get_id(name: str):
    """Extract patient ID from filename (e.g., B1234)"""
    m = re.match(r"(B\d+)", name)
    return m.group(1) if m else None

def find_latest_model():
    """Find the latest fine-tuned model in model/"""
    model_dir = Path(__file__).parent / "model"
    model_runs = sorted([d for d in model_dir.glob(f"{MODEL_SHORT}_*") if d.is_dir()])
    if not model_runs:
        raise FileNotFoundError(f"❌ No model found in {model_dir}")
    latest = model_runs[-1]
    final_model = latest / "final_model"
    if not final_model.exists():
        raise FileNotFoundError(f"❌ No final_model in {latest}")
    print(f"✅ Found model: {final_model}")
    return final_model

def load_data():
    """Load input data for prediction"""
    input_files = sorted(DATA_INPUT.glob("*.txt"))
    output_files = sorted(DATA_OUTPUT.glob("*.txt"))
    
    id2out = {}
    for of in output_files:
        cid = get_id(of.name)
        if cid:
            id2out[cid] = of.read_text(encoding="utf-8").strip()
    
    pairs = []
    for inp_file in input_files:
        cid = get_id(inp_file.name)
        if not cid or cid not in id2out:
            continue
        note = inp_file.read_text(encoding="utf-8").strip()
        pairs.append({"cid": cid, "note": note, "target": id2out[cid]})
    
    return pairs

def predict_all(model_path, pairs):
    """Generate predictions for all pairs"""
    print(f"🔧 Loading model from {model_path}...")
    try:
        tok = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    except Exception as e:
        print(f"❌ Tokenizer load failed: {e}")
        raise
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    
    # Determine output directory based on DATA_OUTPUT name
    data_source = DATA_OUTPUT.name
    pred_dir = Path(__file__).parent / f"predictions_{data_source.replace('data_output_', '').lower()}"
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear old predictions
    for f in pred_dir.glob("*.txt"):
        f.unlink()
    
    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tok,
        device=0 if torch.cuda.is_available() else -1,
        return_full_text=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        dtype=torch.bfloat16,
    )
    
    num_to_predict = len(pairs) if MAX_PRED <= 0 else min(MAX_PRED, len(pairs))
    print(f"🔮 Generating {num_to_predict}/{len(pairs)} predictions...")
    
    for i, pair in enumerate(pairs[:num_to_predict], 1):
        prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user", "content": pair["note"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        out = gen(prompt, max_new_tokens=500, do_sample=False)
        txt = out[0]["generated_text"].strip()
        (pred_dir / f"{pair['cid']}_pred.txt").write_text(txt, encoding="utf-8")
        if i % 10 == 0 or i == num_to_predict:
            print(f"✅ {i}/{num_to_predict} → {pair['cid']}")
    
    print(f"✅ Predictions saved to {pred_dir}")

def main():
    print("=" * 60)
    print("LLAMA 3.1-8B - GENERATE ALL PREDICTIONS")
    print("=" * 60)
    print(f"📥 DATA_INPUT: {DATA_INPUT}")
    print(f"📤 DATA_OUTPUT: {DATA_OUTPUT}")
    
    try:
        model_path = find_latest_model()
        pairs = load_data()
        print(f"📊 Loaded {len(pairs)} patient pairs")
        
        predict_all(model_path, pairs)
        print("\n🎯 PREDICTION PIPELINE COMPLETED")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
