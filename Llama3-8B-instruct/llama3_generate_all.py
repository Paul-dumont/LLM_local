# llama3_generate_all.py
# ==================== LLAMA 3 8B - BATCH INFERENCE ====================
from pathlib import Path
import os, re, time, json, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import Dataset
from huggingface_hub import login

# ==================== HF LOGIN ====================
hf_token = os.getenv("HF_TOKEN", None)
if hf_token:
    login(token=hf_token, add_to_git_credential=False)
    print(f"✅ HuggingFace token loaded from environment")
else:
    print("⚠️ No HF_TOKEN in environment, will try to use cached token")

# ==================== PARAMS ====================
MODEL_ID    = "meta-llama/Llama-3-8b-Instruct"  # ✅ Meta Llama 3 8B Instruct
MODEL_SHORT = "llama3_8B_gen"

MAX_SEQ_LEN  = int(os.getenv("MAX_SEQ_LEN", 4096))
MAX_INPUTS   = int(os.getenv("MAX_INPUTS", "0"))      # 0 = all
SEED         = int(os.getenv("SEED", "42"))
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", 4))        # batch for inference

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Adapte à ton arborescence
BASE_DIR     = Path(__file__).parent.parent
TRAINING_DIR = BASE_DIR / "data_training"
DATA_INPUT   = TRAINING_DIR / "data_input"
OUTPUT_DIR   = Path(__file__).parent / "predictions_all"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"🏠 BASE_DIR: {BASE_DIR}")
print(f"📥 DATA_INPUT: {DATA_INPUT}")
print(f"💾 OUTPUT_DIR: {OUTPUT_DIR}")

INSTRUCTION = "Using the following note, extract structured key-value pairs about the patient's symptoms and diagnoses:"

# ==================== UTILS ====================
def get_id(name: str):
    m = re.match(r"(B\d+)", name)
    return m.group(1) if m else None

def setup_tokenizer():
    print("🔧 Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"

    def render(messages, add_generation_prompt=True):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return tok, render

def load_data():
    print("📂 Loading all input data...")
    input_files = sorted(DATA_INPUT.glob("*.txt"))
    print(f"📁 Found {len(input_files)} input files")

    files_to_process = input_files if MAX_INPUTS == 0 else input_files[:MAX_INPUTS]

    pairs = []
    for inp_file in files_to_process:
        cid = get_id(inp_file.name)
        if not cid:
            print(f"❌ No ID found for: {inp_file.name}")
            continue

        note = inp_file.read_text(encoding="utf-8").strip()
        pairs.append({"cid": cid, "note": note})

    print(f"✅ Loaded {len(pairs)} samples")
    return pairs

def load_model(tok):
    print("🤖 Loading model (bf16)…")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.use_cache = True
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model

def predict_batch(model, tok, pairs, render):
    """Generate predictions for all samples using pipeline"""
    print(f"🔮 Generating predictions for {len(pairs)} samples...")
    
    gen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tok,
        device=0 if torch.cuda.is_available() else -1,
        return_full_text=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        dtype=torch.bfloat16,
        batch_size=BATCH_SIZE,
    )

    for i, pair in enumerate(pairs, 1):
        prompt = render(
            [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user",   "content": pair["note"]},
            ],
            add_generation_prompt=True,
        )
        
        out = gen(prompt, max_new_tokens=500, do_sample=False)
        txt = out[0]["generated_text"].strip()
        (OUTPUT_DIR / f"{pair['cid']}_pred.txt").write_text(txt, encoding="utf-8")
        
        if i % 10 == 0 or i == len(pairs):
            print(f"✅ Pred {i}/{len(pairs)} → {pair['cid']}")

def main():
    print(f"🏠 OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"⏱️  Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Clear previous predictions
    for f in OUTPUT_DIR.glob("*.txt"):
        f.unlink()
    
    tok, render = setup_tokenizer()
    pairs = load_data()
    model = load_model(tok)
    predict_batch(model, tok, pairs, render)
    
    print(f"\n🎯 BATCH PREDICTION COMPLETED")
    print(f"⏱️  End time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📍 Predictions saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
