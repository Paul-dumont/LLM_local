# llama_full_ft.py
# ==================== LLAMA 3.1-8B-INSTRUCT - FULL FINETUNE (1x GPU, NO LoRA/NO QUANT) ====================
from pathlib import Path
import os, re, time, json, random
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# ==================== PARAMS ====================
# Allow overriding the model from environment. Default to an open model if no HF token is available.
# Use an open, non-gated model by default to avoid 401 errors. You can override by setting MODEL_ID env var.
MODEL_ID    = os.getenv("MODEL_ID", "microsoft/Phi-3.5-mini-instruct")
MODEL_SHORT = os.getenv("MODEL_SHORT", "phi_3.5_mini")

# DEBUG: Print the model being used
print(f"🔍 MODEL_ID env var: {os.getenv('MODEL_ID', 'NOT_SET')}")
print(f"✅ Using MODEL_ID: {MODEL_ID}")

# Optimisé pour 44GB VRAM - Phi-3.5 Mini (3.8B - très léger)
MAX_SEQ_LEN  = int(os.getenv("MAX_SEQ_LEN", 4096))    # Gardé à 4096
NUM_EPOCHS   = int(os.getenv("NUM_EPOCHS", 3))
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", 2))        # 2 per-device (Phi est léger!)
GRAD_ACCUM   = int(os.getenv("GRAD_ACCUM", 4))        # 4 (effective BS = 2*4 = 8)
LR           = float(os.getenv("LR", "2e-5"))         # 2e-5 (Phi plus stable)
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.10"))
SAVE_STEPS   = int(os.getenv("SAVE_STEPS", 200))
EVAL_STEPS   = int(os.getenv("EVAL_STEPS", 200))
LOG_STEPS    = int(os.getenv("LOG_STEPS", 50))
USE_WANDB    = os.getenv("USE_WANDB", "0") == "1"
EVAL_RATIO   = float(os.getenv("EVAL_RATIO", "0.2"))
MAX_INPUTS   = int(os.getenv("MAX_INPUTS", "0"))      # 0 = all
MAX_PRED     = int(os.getenv("MAX_PRED", "0"))
SEED         = int(os.getenv("SEED", "42"))

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Adapte à ton arborescence - lancé depuis /home/luciacev/Desktop/LLM
# Structure: LLM/data_training/{data_input, data_output_clean}
#            LLM/model/...
BASE_DIR     = Path(__file__).parent.parent  # /nas/longleaf/home/dumontp/LONGLEAF/LLM_local
TRAINING_DIR = BASE_DIR / "data_training"
DATA_INPUT   = TRAINING_DIR / "data_input"
DATA_OUTPUT  = TRAINING_DIR / "data_output_clean"
MODEL_DIR    = Path(__file__).parent / "Phi-3.5-mini_model"  # Sauvegarde dans Phi-3.5-mini_model/
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR      = MODEL_DIR / f"{MODEL_SHORT}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getenv('SLURM_JOB_ID', 'local')}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

print(f"🏠 BASE_DIR: {BASE_DIR}")
print(f"📂 TRAINING_DIR: {TRAINING_DIR}")
print(f"📥 DATA_INPUT: {DATA_INPUT}")
print(f"📤 DATA_OUTPUT: {DATA_OUTPUT}")
print(f"💾 RUN_DIR: {RUN_DIR}")

INSTRUCTION = "Using the following note, extract structured key-value pairs about the patient's symptoms and diagnoses:"

# ==================== UTILS ====================
def get_id(name: str):
    # Extrait un identifiant commun (ex: B1234) du nom de fichier
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

    # Check if model has chat template support
    has_chat_template = hasattr(tok, 'chat_template') and tok.chat_template is not None
    
    def render(messages, add_generation_prompt=True):
        """Render messages with chat template if available, else fallback to simple concatenation"""
        if has_chat_template:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        else:
            # Fallback: manually build the text
            text = ""
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    text += f"System: {content}\n\n"
                elif role == "user":
                    text += f"User: {content}\n\n"
                elif role == "assistant":
                    text += f"Assistant: {content}\n\n"
            return text.rstrip()
    
    return tok, render

def load_data(tok):
    print("📂 Loading all available data...")
    input_files  = sorted(DATA_INPUT.glob("*.txt"))
    output_files = sorted(DATA_OUTPUT.glob("*.txt"))
    print(f"📁 Found {len(input_files)} input files and {len(output_files)} output files")

    files_to_process = input_files if MAX_INPUTS == 0 else input_files[:MAX_INPUTS]

    id2out = {}
    for of in output_files:
        cid = get_id(of.name)
        if cid:
            id2out[cid] = of
    print(f"📁 id2out has {len(id2out)} entries")

    pairs = []
    for inp_file in files_to_process:
        cid = get_id(inp_file.name)
        if not cid:
            print(f"❌ No ID found for: {inp_file.name}")
            continue
        if cid not in id2out:
            print(f"❌ No output file found for {cid}")
            continue

        note   = inp_file.read_text(encoding="utf-8").strip()
        target = id2out[cid].read_text(encoding="utf-8").strip()

        messages = [
            {"role": "system",    "content": INSTRUCTION},
            {"role": "user",      "content": note},
            {"role": "assistant", "content": target},
        ]
        pairs.append({"cid": cid, "messages": messages, "note_only": note})

    random.shuffle(pairs)
    n_eval = max(1, int(len(pairs) * EVAL_RATIO)) if pairs else 0
    eval_pairs  = pairs[:n_eval]
    train_pairs = pairs[n_eval:]
    print(f"✅ Pairs: total={len(pairs)} | train={len(train_pairs)} | eval={len(eval_pairs)}")
    return pairs, train_pairs, eval_pairs

def make_datasets(train_pairs, eval_pairs, render):
    train_texts = [render(p["messages"], add_generation_prompt=False) for p in train_pairs]
    eval_texts  = [render(p["messages"], add_generation_prompt=False) for p in eval_pairs]
    dset = DatasetDict({
        "train": Dataset.from_list([{"text": t} for t in train_texts]),
        "eval":  Dataset.from_list([{"text": t} for t in eval_texts]),
    })
    print(f"✅ Datasets ready: train={len(dset['train'])} eval={len(dset['eval'])}")
    return dset

def load_model(tok):
    print("🤖 Loading base model for full finetune (bf16)…")
    
    # Phi-3.5 needs eager attention, others can use sdpa
    attn_impl = "eager" if "phi" in MODEL_ID.lower() else "sdpa"
    print(f"   Using attention implementation: {attn_impl}")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.use_cache = False  # disable during training
    # memory helpers
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model

def sft_config():
    report_to = ["wandb"] if USE_WANDB else []
    return SFTConfig(
        output_dir=str(RUN_DIR),
        logging_dir=str(RUN_DIR / "logs"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=max(1, min(2, BATCH_SIZE)),
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        eval_strategy="steps",     # <- corrige "eval_strategy"
        eval_steps=EVAL_STEPS,
        logging_strategy="steps",
        logging_steps=LOG_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=1,
        load_best_model_at_end=False,    # évite un 2e chargement en mémoire
        fp16=False,
        bf16=True,                       # A100/L40S : OK
        gradient_checkpointing=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=0,
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
        dataset_text_field="text",
        report_to=report_to,
        run_name=f"{MODEL_SHORT}_{int(time.time())}" if USE_WANDB else None,
        # Optimiseur VRAM-friendly: adafactor économise mémoire sans dépendre de bitsandbytes
        optim="adafactor",
        adam_epsilon=1e-8,
    )

def train_and_eval(model, tok, dset, cfg):
    print("🏋️ Training…")
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dset["train"],
        eval_dataset=dset["eval"],
        tokenizer=tok,
    )
    tr = trainer.train()
    print("✅ Training finished")
    results = {
        "training_loss": float(tr.training_loss) if tr.training_loss is not None else None,
        "global_step": tr.global_step,
        "epochs": NUM_EPOCHS,
    }
    (RUN_DIR / "training_results.json").write_text(json.dumps(results, indent=2))
    print(f"📄 Saved: {RUN_DIR/'training_results.json'}")

    print("🧪 Evaluating best model…")
    ev = trainer.evaluate()
    (RUN_DIR / "eval_results.json").write_text(json.dumps(ev, indent=2))
    print(f"📄 Saved: {RUN_DIR/'eval_results.json'}")
    return trainer

def save_final(trainer, tok):
    out = RUN_DIR / "final_model"
    out.mkdir(exist_ok=True, parents=True)
    print("💾 Saving final model/tokenizer…")
    trainer.model.save_pretrained(str(out))
    tok.save_pretrained(str(out))
    print(f"✅ Saved to {out}")

def predict_samples(trainer, tok, eval_pairs, render):
    """Generate predictions for evaluation set (unseen patients only)"""
    if len(eval_pairs) == 0:
        print("⚠️ No evaluation pairs to predict")
        return
    
    try:
        # Dossier de prédictions : predict_phi_eval
        pred_dir = Path(__file__).parent / "predict_phi_eval"
        pred_dir.mkdir(parents=True, exist_ok=True)
        for f in pred_dir.glob("*.txt"):
            f.unlink()

        model = trainer.model.eval()
        
        # Free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        num_to_predict = len(eval_pairs) if MAX_PRED <= 0 else min(MAX_PRED, len(eval_pairs))
        print(f"🔮 Generating {num_to_predict} predictions for unseen patients...")
        
        with torch.no_grad():
            for i, pair in enumerate(eval_pairs[:num_to_predict], 1):
                try:
                    prompt = render(
                        [
                            {"role": "system", "content": INSTRUCTION},
                            {"role": "user",   "content": pair["note_only"]},
                        ],
                        add_generation_prompt=True,
                    )
                    
                    # Tokenize without pipeline to avoid compatibility issues
                    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
                    if torch.cuda.is_available():
                        inputs = inputs.to("cuda")
                    
                    # Generate with use_cache=False to avoid DynamicCache issues
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=500,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id or tok.eos_token_id,
                        use_cache=False,  # Disable cache to avoid compatibility issues
                    )
                    
                    # Decode
                    txt = tok.decode(output_ids[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
                    (pred_dir / f"{pair['cid']}_pred.txt").write_text(txt, encoding="utf-8")
                    
                    if i % 10 == 0 or i == num_to_predict:
                        print(f"✅ Pred {i}/{num_to_predict} → {pair['cid']}")
                
                except Exception as e:
                    print(f"❌ Error predicting for {pair.get('cid', 'unknown')} (item {i}): {e}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    except Exception as e:
        print(f"❌ Error in predict_samples: {e}")
        import traceback
        traceback.print_exc()
        raise

def main():
    print(f"🏠 RUN_DIR: {RUN_DIR}")
    try:
        tok, render = setup_tokenizer()
        pairs, train_pairs, eval_pairs = load_data(tok)
        dset = make_datasets(train_pairs, eval_pairs, render)
        model = load_model(tok)
        cfg = sft_config()

        # persist config
        (RUN_DIR / "training_config.json").write_text(
            json.dumps(
                {
                    "model_id": MODEL_ID,
                    "max_seq_len": MAX_SEQ_LEN,
                    "epochs": NUM_EPOCHS,
                    "batch_size": BATCH_SIZE,
                    "grad_accum": GRAD_ACCUM,
                    "lr": LR,
                    "weight_decay": WEIGHT_DECAY,
                    "warmup_ratio": WARMUP_RATIO,
                    "sizes": {
                        "total_pairs": len(pairs),
                        "train": len(train_pairs),
                        "eval": len(eval_pairs),
                    },
                },
                indent=2,
            )
        )

        trainer = train_and_eval(model, tok, dset, cfg)
        save_final(trainer, tok)
        print("🔮 Starting predictions...")
        predict_samples(trainer, tok, eval_pairs, render)
        print("\n🎯 FULL TRAINING PIPELINE COMPLETED")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
