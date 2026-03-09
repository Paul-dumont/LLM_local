# qwen_dryrun.py
# ==================== QWEN 2.5-7B INSTRUCT - DRY RUN PREDICTION (NO TRAINING) ====================
# Charge le modèle de base et prédit sur TOUS les patients (500) sans fine-tuning
from pathlib import Path
import os, re, time, json, random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm

# ==================== PARAMS ====================
MODEL_ID    = "Qwen/Qwen2.5-7B-Instruct"
MODEL_SHORT = "qwen_dryrun_7B"

MAX_SEQ_LEN  = int(os.getenv("MAX_SEQ_LEN", 4096))
NUM_PATIENTS = 10                                       # ⬅️ CHANGE ICI: 5, 100, 500, etc.
MAX_INPUTS   = int(os.getenv("MAX_INPUTS", "0"))      # 0 = all
MAX_PRED     = int(os.getenv("MAX_PRED", "0"))
SEED         = int(os.getenv("SEED", "42"))

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==================== PATHS ====================
BASE_DIR     = Path(__file__).parent.parent  # /home/luciacev/Desktop/LLM
TRAINING_DIR = BASE_DIR / "data_training"
DATA_INPUT   = TRAINING_DIR / "data_input"
MODEL_DIR    = Path(__file__).parent / "model"
PRED_DIR     = Path(__file__).parent / "predictions_baseline"
PRED_DIR.mkdir(parents=True, exist_ok=True)

print(f"🏠 BASE_DIR: {BASE_DIR}")
print(f"📥 DATA_INPUT: {DATA_INPUT}")
print(f"📤 PRED_DIR: {PRED_DIR}")

INSTRUCTION = """Extract structured patient information from the clinical note in key-value pair format.

OUTPUT FORMAT: One field per line as "field_name: value"
Use pipe (|) to separate multiple values.
Use "unknown" if information is not mentioned in the note.

FIELD EXTRACTION GUIDE WITH EXAMPLES:

IDENTIFICATION:
- patient_id: Extract patient identifier (e.g., "B001")
- patient_age: Extract age in years (e.g., "26")

PAIN SYMPTOMS:
- average_daily_pain_intensity: Rate 0-10 (e.g., "8")
- tmj_pain_rating: TMJ pain intensity (e.g., "6")
- headache_intensity: Headache severity (e.g., "6 | present")
- pain_onset_date: When pain started (e.g., "4 months ago")
- pain_duration: How long pain has lasted (e.g., "unknown")
- pain_frequency: How often pain occurs (e.g., "unknown")
- pain_aggravating_factors: What makes it worse (e.g., "clenching | bruxism")
- pain_relieving_factors: What helps pain (e.g., "unknown")

JAW MEASUREMENTS & FUNCTION:
- maximum_opening: Maximum jaw opening in mm (e.g., "49mm")
- maximum_opening_without_pain: Pain-free opening in mm (e.g., "30mm")
- jaw_function_score: Functional rating (e.g., "5")
- jaw_clicking: Present/absent (e.g., "unknown")
- jaw_locking: Present/absent (e.g., "unknown")
- jaw_crepitus: Grinding sound present (e.g., "unknown")

HEAD & NECK SYMPTOMS:
- headache_location: Where headaches occur (e.g., "frontal | temporal | posterior head | behind her eyes")
- headache_frequency: How often headaches occur (e.g., "unknown")
- neck_pain_present: true/false/unknown (e.g., "true")
- back_pain_present: true/false/unknown (e.g., "true")
- earache_present: true/false/unknown (e.g., "true")
- tinnitus_present: true/false/unknown (e.g., "true")
- vertigo_present: true/false/unknown (e.g., "true")
- hearing_loss_present: true/false/unknown (e.g., "true")

SLEEP & BREATHING:
- sleep_disorder_type: Types of sleep issues (e.g., "snoring | daytime tiredness | mouth breathing")
- sleep_apnea_diagnosed: true/false/unknown (e.g., "false")
- airway_obstruction_present: true/false/unknown (e.g., "unknown")

DIAGNOSES & CONDITIONS:
- disc_displacement: Location and type (e.g., "left disc anterior displacement with early reduction dislocation | right disc anterior displacement with early reduction")
- joint_arthritis_location: Where arthritis is found (e.g., "left tmj | right tmj")
- fibromyalgia_present: true/false/unknown (e.g., "unknown")
- autoimmune_condition: Any autoimmune disease (e.g., "unknown")
- depression_present: true/false/unknown (e.g., "unknown")
- migraine_history: Migraine history (e.g., "unknown")

MUSCLE & PAIN LOCATION:
- muscle_pain_location: Where muscles hurt (e.g., "head | neck")
- muscle_pain_score: Severity rating (e.g., "moderate to high moderate")
- muscle_symptoms_present: true/false/unknown (e.g., "unknown")
- joint_pain_areas: All areas with joint pain (e.g., "tmj | left tmj | left hip | knees | knees | back")

TREATMENTS & HISTORY:
- current_medications: Medications taken (e.g., "unknown")
- previous_medications: Past medications (e.g., "unknown")
- current_appliance: Current dental/medical device (e.g., "unknown")
- appliance_history: History of appliances used (e.g., "unknown")
- physical_therapy_status: PT history (e.g., "unknown")
- onset_triggers: What caused the onset (e.g., "mva hitting her whole left side")

OTHER SYMPTOMS:
- diet_score: Diet quality rating (e.g., "3")
- disability_rating: Functional disability level (e.g., "2")
- adverse_reactions: Medication reactions (e.g., "unknown")

INSTRUCTIONS:
1. Extract ONLY information explicitly mentioned in the clinical note
2. Output one field per line in the exact format shown
3. For boolean-like fields (present/true/false), use: "true", "false", or "unknown"
4. For multiple values, separate with " | " (space-pipe-space)
5. Keep values concise but complete
6. Use exact wording from the note when possible
7. If a field is not mentioned, use "unknown"
8. Do NOT add explanations or extra text - only the key-value pairs"""

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

    def render(messages, add_generation_prompt=True):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return tok, render

def load_input_files():
    """Charge TOUS les fichiers d'entrée (500 patients)"""
    print("📂 Loading all input files...")
    input_files = sorted(DATA_INPUT.glob("*.txt"))
    print(f"📁 Found {len(input_files)} input files")
    
    files_to_process = input_files[:NUM_PATIENTS]
    print(f"📁 Processing {len(files_to_process)} files")
    
    patients = []
    for inp_file in files_to_process:
        cid = get_id(inp_file.name)
        if not cid:
            print(f"❌ No ID found for: {inp_file.name}")
            continue
        
        note = inp_file.read_text(encoding="utf-8").strip()
        patients.append({
            "cid": cid,
            "note": note,
        })
    
    print(f"✅ Loaded {len(patients)} patient notes")
    return patients

def load_model(tok):
    print("🤖 Loading base model for inference (bf16)…")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    model.config.use_cache = True  # enable for inference
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model

def predict_all(model, tok, patients, render):
    """Génère les prédictions pour TOUS les patients"""
    # Clear previous predictions
    for f in PRED_DIR.glob("*.txt"):
        f.unlink()
    
    gen = pipeline(
        task="text-generation",
        model=model.eval(),
        tokenizer=tok,
        device=0 if torch.cuda.is_available() else -1,
        return_full_text=False,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        dtype=torch.bfloat16,
    )

    num_to_predict = len(patients) if MAX_PRED <= 0 else min(MAX_PRED, len(patients))
    print(f"🔮 Generating {num_to_predict} predictions on base model (baseline)...")
    print(f"📁 Predictions will be saved to: {PRED_DIR}\n")
    
    start_time = time.time()
    
    pbar = tqdm(patients[:num_to_predict], desc="Predictions", unit="patient")
    for i, patient in enumerate(pbar, 1):
        prompt = render(
            [
                {"role": "system", "content": INSTRUCTION},
                {"role": "user",   "content": patient["note"]},
            ],
            add_generation_prompt=True,
        )
        
        out = gen(prompt, max_new_tokens=500, do_sample=False)
        txt = out[0]["generated_text"].strip()
        (PRED_DIR / f"{patient['cid']}_pred.txt").write_text(txt, encoding="utf-8")
        
        elapsed = time.time() - start_time
        rate = elapsed / i
        eta = rate * (num_to_predict - i)
        pbar.set_postfix({
            "ID": patient['cid'],
            "Elapsed": f"{elapsed:.1f}s",
            "ETA": f"{eta:.0f}s"
        })

def save_metadata():
    """Sauvegarde les métadonnées de la run"""
    metadata = {
        "model_id": MODEL_ID,
        "model_type": "baseline_no_finetuning",
        "max_seq_len": MAX_SEQ_LEN,
        "inference_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prediction_dir": str(PRED_DIR),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    metadata_file = PRED_DIR / "metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))
    print(f"📄 Metadata saved: {metadata_file}")

def main():
    print(f"\n{'='*70}")
    print(f"🚀 QWEN 7B BASELINE PREDICTION (DRY RUN - NO FINE-TUNING)")
    print(f"{'='*70}\n")
    
    tok, render = setup_tokenizer()
    patients = load_input_files()
    model = load_model(tok)
    predict_all(model, tok, patients, render)
    save_metadata()
    
    print(f"\n{'='*70}")
    print(f"✨ BASELINE PREDICTION COMPLETED")
    print(f"📁 Predictions saved in: {PRED_DIR}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
