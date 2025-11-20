#!/usr/bin/env python3
"""
Semantic Similarity Evaluation - Using Average Distances
Compare Data_output_12 (references) vs predictions_12 (model outputs)
Shows average semantic similarity scores (0-1) for each criterion
No F1 score - just pure semantic distances
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

# Try to import semantic model
try:
    from sentence_transformers import SentenceTransformer, util
    print("✅ Loading Semantic Model: all-MiniLM-L6-v2")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    SEMANTIC_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Semantic model not available: {e}")
    SEMANTIC_AVAILABLE = False

# === Configuration (top-level) ===
# Reference and prediction folders can be adjusted here. Use REPO_ROOT so paths are repo-relative.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Model configuration (short name used in prediction folder names)
# Set `model` to the short model name (example: 'Qwen1.5B', 'Mistral7B', 'Mistral7B')
model = 'Qwen1.5B'
model_folder = f"{model}-instruct"

# Reference and prediction folders (repo-relative)
REF_FOLDER = str(REPO_ROOT / 'data_predicition' / 'data_output_harmonized')
PRED_FOLDER = str(REPO_ROOT / model_folder / f"predict_{model}_eval_harmonized")

# Define the 46 criteria - sorted by representation rate (highest to lowest)
CRITERIA = [
    "fibromyalgia_present",           # 97.60%
    "pain_duration",                  # 94.80%
    "depression_present",             # 93.40%
    "pain_frequency",                 # 92.20%
    "previous_medications",           # 87.80%
    "pain_relieving_factors",         # 86.60%
    "current_medications",            # 85.80%
    "pain_aggravating_factors",       # 84.60%
    "headache_frequency",             # 81.60%
    "patient_age",                    # 81.20%
    "headache_intensity",             # 79.80%
    "onset_triggers",                 # 78.80%
    "headache_location",              # 78.40%
    "back_pain_present",              # 77.20%
    "neck_pain_present",              # 75.60%
    "average_daily_pain_intensity",   # 75.20%
    "migraine_history",               # 72.80%
    "jaw_locking",                    # 72.20%
    "adverse_reactions",              # 70.40%
    "muscle_pain_score",              # 68.80%
    "sleep_disorder_type",            # 68.40%
    "jaw_clicking",                   # 68.00%
    "muscle_symptoms_present",        # 66.80%
    "tmj_pain_rating",                # 65.60%
    "tinnitus_present",               # 63.80%
    "vertigo_present",                # 62.80%
    "disc_displacement",              # 61.60%
    "physical_therapy_status",        # 59.00%
    "current_appliance",              # 58.60%
    "appliance_history",              # 58.20%
    "hearing_loss_present",           # 57.60%
    "joint_pain_areas",               # 55.80%
    "muscle_pain_location",           # 54.80%
    "joint_arthritis_location",       # 53.20%
    "earache_present",                # 50.40%
    "jaw_crepitus",                   # 48.60%
    "disability_rating",              # 47.80%
    "jaw_function_score",             # 45.60%
    "sleep_apnea_diagnosed",          # 45.00%
    "airway_obstruction_present",     # 42.40%
    "diet_score",                     # 39.40%
    "maximum_opening",                # 25.00%
    "maximum_opening_without_pain",   # 20.20%
    "pain_onset_date",                # 11.00%
    "autoimmune_condition",           # 8.20%
    "patient_id"                      # 0.00%
]

# Optional: explicit ordering by semantic representation/rate (highest → lowest)
SEMANTIC_ORDER = [
    "patient_id",
    "patient_age",
    "headache_intensity",
    "tmj_pain_rating",
    "disc_displacement",
    "joint_arthritis_location",
    "jaw_function_score",
    "maximum_opening",
    "diet_score",
    "disability_rating",
    "tinnitus_present",
    "vertigo_present",
    "joint_pain_areas",
    "earache_present",
    "pain_aggravating_factors",
    "average_daily_pain_intensity",
    "airway_obstruction_present",
    "pain_onset_date",
    "appliance_history",
    "current_medications",
    "headache_location",
    "muscle_pain_location",
    "muscle_symptoms_present",
    "muscle_pain_score",
    "hearing_loss_present",
    "jaw_clicking",
    "headache_frequency",
    "sleep_disorder_type",
    "maximum_opening_without_pain",
    "neck_pain_present",
    "current_appliance",
    "onset_triggers",
    "physical_therapy_status",
    "adverse_reactions",
    "jaw_crepitus",
    "jaw_locking",
    "pain_relieving_factors",
    "back_pain_present",
    "sleep_apnea_diagnosed",
    "autoimmune_condition",
    "migraine_history",
    "previous_medications",
    "pain_frequency",
    "depression_present",
    "pain_duration",
    "fibromyalgia_present",
]

UNKNOWN_VARIATIONS = ["unknown", "unknow", "n/a", "na", "not available", "not provided", "none", ""]


def is_unknown(value):
    """Check if value is an unknown/empty value"""
    normalized = value.lower().strip()
    return normalized in UNKNOWN_VARIATIONS or normalized == ""


def semantic_similarity_distance(text1, text2):
    """
    Calculate semantic similarity distance between two texts.
    Returns a value between 0 and 1 where:
    - 1.0 = identical meaning
    - 0.0 = completely different
    """
    if not SEMANTIC_AVAILABLE:
        # Fallback: exact match
        return 1.0 if text1.lower().strip() == text2.lower().strip() else 0.0
    
    if not text1 or not text2:
        # Both empty or one empty
        return 1.0 if text1 == text2 else 0.0
    
    try:
        embeddings1 = semantic_model.encode(text1, convert_to_tensor=True)
        embeddings2 = semantic_model.encode(text2, convert_to_tensor=True)
        similarity = float(util.pytorch_cos_sim(embeddings1, embeddings2)[0][0])
        # Clamp between 0 and 1
        return max(0.0, min(1.0, similarity))
    except:
        return 1.0 if text1.lower().strip() == text2.lower().strip() else 0.0


def extract_criterion_from_file(filepath):
    """Extract criterion values from a prediction/reference file"""
    if not os.path.exists(filepath):
        return {}
    
    values = {}
    
    # Initialize all criteria with empty strings
    for criterion in CRITERIA:
        values[criterion] = ""
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Parse each line by looking for "key: value" format
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                # Only store if it's a known criterion
                if key in CRITERIA:
                    values[key] = value
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    
    return values


def compare_folders(ref_folder, pred_folder, filter_unknown=False):
    """Compare reference and prediction folders using semantic similarity distances"""
    criterion_scores = defaultdict(list)
    criterion_counts = defaultdict(int)
    unknown_stats = defaultdict(lambda: {"total": 0, "unknown_ref": 0, "unknown_pred": 0})
    
    ref_path = Path(ref_folder)
    pred_path = Path(pred_folder)
    
    # Get all prediction files
    pred_files = sorted(pred_path.glob("B*_pred.txt"))
    ref_files = sorted(ref_path.glob("B*_summary.txt"))
    
    print(f"📊 Evaluating {len(pred_files)} patients...\n")
    
    for pred_file in pred_files:
        patient_id = pred_file.name.replace("_pred.txt", "")
        ref_file = ref_path / f"{patient_id}_summary.txt"
        
        if not ref_file.exists():
            continue
        
        # Extract values
        ref_values = extract_criterion_from_file(str(ref_file))
        pred_values = extract_criterion_from_file(str(pred_file))
        
        # Compare each criterion
        for criterion in CRITERIA:
            ref_val = ref_values.get(criterion, "").strip()
            pred_val = pred_values.get(criterion, "").strip()
            
            # Track unknowns
            unknown_stats[criterion]["total"] += 1
            if is_unknown(ref_val):
                unknown_stats[criterion]["unknown_ref"] += 1
            if is_unknown(pred_val):
                unknown_stats[criterion]["unknown_pred"] += 1
            
            # Skip if both empty
            if not ref_val and not pred_val:
                continue
            
            # Skip if filtering unknowns and either is unknown
            if filter_unknown and (is_unknown(ref_val) or is_unknown(pred_val)):
                continue
            
            # Calculate semantic similarity
            similarity = semantic_similarity_distance(ref_val, pred_val)
            criterion_scores[criterion].append(similarity)
            criterion_counts[criterion] += 1
    
    return criterion_scores, criterion_counts, unknown_stats


def main():
   
    if not SEMANTIC_AVAILABLE:
        print("❌ Semantic model not available. Install with:")
        print("   pip install sentence-transformers")
        return
    
    # Use ONLY the harmonized data_output as the manual/reference folder
    manual_folder = str(REPO_ROOT / 'data_predicition' / 'data_output_harmonized')

    # llm_folder is derived from the top-level `model` and `model_folder` variables
    llm_folder = str(REPO_ROOT / model_folder / f"predict_{model}_eval_harmonized")

    # Keep backward-compatible names used later in the script
    ref_folder = manual_folder
    pred_folder = llm_folder
    # pred_folder = "/home/luciacev/Desktop/LLM/Mistral-7B-instruct/predict_Mistral7B_eval"

    
    if not os.path.exists(ref_folder):
        print(f"❌ Reference folder not found: {ref_folder}")
        return
    
    if not os.path.exists(pred_folder):
        print(f"❌ Prediction folder not found: {pred_folder}")
        return
    
    print(f"\n📁 References: {ref_folder}")
    print(f"📁 Predictions: {pred_folder}\n")
    
    # ===== MODE 1: With unknowns =====
    
    scores_normal, counts_normal, unknown_stats = compare_folders(
        ref_folder, pred_folder, filter_unknown=False
    )
    
    # Calculate statistics
    criterion_stats_normal = {}
    for criterion in CRITERIA:
        scores = scores_normal.get(criterion, [])
        if scores:
            mean = np.mean(scores)
            std = np.std(scores)
            count = len(scores)
        else:
            mean = 0.0
            std = 0.0
            count = 0
        
        criterion_stats_normal[criterion] = {
            'mean': mean,
            'std': std,
            'count': count
        }
    
    # Overall statistics
    all_scores_normal = []
    for scores in scores_normal.values():
        all_scores_normal.extend(scores)
    
    overall_mean_normal = np.mean(all_scores_normal) if all_scores_normal else 0
    overall_std_normal = np.std(all_scores_normal) if all_scores_normal else 0
        
    # ===== MODE 2: Without unknowns =====

    
    scores_filtered, counts_filtered, _ = compare_folders(
        ref_folder, pred_folder, filter_unknown=True
    )
    
    criterion_stats_filtered = {}
    for criterion in CRITERIA:
        scores = scores_filtered.get(criterion, [])
        if scores:
            mean = np.mean(scores)
            std = np.std(scores)
            count = len(scores)
        else:
            mean = 0.0
            std = 0.0
            count = 0
        
        criterion_stats_filtered[criterion] = {
            'mean': mean,
            'std': std,
            'count': count
        }
    
    all_scores_filtered = []
    for scores in scores_filtered.values():
        all_scores_filtered.extend(scores)
    
    overall_mean_filtered = np.mean(all_scores_filtered) if all_scores_filtered else 0
    overall_std_filtered = np.std(all_scores_filtered) if all_scores_filtered else 0
    
    
    # ===== COMPARISON =====
    print(f"\n{'Metric':<30} {'With Unknowns':<20} {'Without Unknowns':<20}")
    print("-"*70)
    print(f"{'Mean':<30} {overall_mean_normal:.4f}{' '*15} {overall_mean_filtered:.4f}")
    print(f"{'Std Dev':<30} {overall_std_normal:.4f}{' '*15} {overall_std_filtered:.4f}")
    print(f"{'Total comparisons':<30} {len(all_scores_normal):<20} {len(all_scores_filtered):<20}")
    
    # ===== PER-CRITERION COMPARISON =====
  
    print(f"\n{'Criterion':<40} {'With UK':<15} {'No UK':<15}")
    print("-"*70)
    
    # Build a dict of per-criterion stats (kept for easy lookup)
    criterion_diffs = {}
    for criterion in CRITERIA:
        normal = criterion_stats_normal[criterion]['mean']
        filtered = criterion_stats_filtered[criterion]['mean']
        criterion_diffs[criterion] = {
            'normal': normal,
            'filtered': filtered
        }

    # Print all criteria in the semantic order provided by SEMANTIC_ORDER
    # (highest → lowest semantic representation as provided by the user)
    print("\n📊 ALL CRITERIA (ordered by semantic presence: HIGH → LOW):")
    for criterion in SEMANTIC_ORDER:
        stats = criterion_diffs.get(criterion, {'normal': 0.0, 'filtered': 0.0})
        print(f"{criterion:<40} {stats['normal']:.4f}          {stats['filtered']:.4f}")
    



if __name__ == "__main__":
    main()
