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

# Define the 46 criteria
CRITERIA = [
    "patient_id",
    "maximum_opening_without_pain",
    "patient_age",
    "maximum_opening",
    "pain_relieving_factors",
    "diet_score",
    "pain_duration",
    "disability_rating",
    "sleep_apnea_diagnosed",
    "jaw_function_score",
    "tinnitus_present",
    "vertigo_present",
    "hearing_loss_present",
    "back_pain_present",
    "neck_pain_present",
    "pain_aggravating_factors",
    "physical_therapy_status",
    "earache_present",
    "onset_triggers",
    "adverse_reactions",
    "fibromyalgia_present",
    "jaw_locking",
    "airway_obstruction_present",
    "depression_present",
    "headache_frequency",
    "pain_frequency",
    "muscle_symptoms_present",
    "jaw_clicking",
    "previous_medications",
    "muscle_pain_score",
    "migraine_history",
    "headache_location",
    "current_appliance",
    "average_daily_pain_intensity",
    "headache_intensity",
    "autoimmune_condition",
    "pain_onset_date",
    "jaw_crepitus",
    "tmj_pain_rating",
    "sleep_disorder_type",
    "current_medications",
    "disc_displacement",
    "muscle_pain_location",
    "joint_arthritis_location",
    "appliance_history",
    "joint_pain_areas"
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
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, criterion in enumerate(CRITERIA):
            if i < len(lines):
                line = lines[i].strip()
                # Parse "key: value" format
                if ':' in line:
                    value = line.split(':', 1)[1].strip()
                else:
                    value = line
                values[criterion] = value
            else:
                values[criterion] = ""
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
    
    print(f"\n📊 Found {len(ref_files)} references, {len(pred_files)} predictions")
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
    print("\n" + "="*120)
    print("🧠 SEMANTIC SIMILARITY EVALUATION (Average Distances)")
    print("Data_output_12 vs predictions_12")
    print("="*120)
    
    if not SEMANTIC_AVAILABLE:
        print("❌ Semantic model not available. Install with:")
        print("   pip install sentence-transformers")
        return
    
    ref_folder = "/home/luciacev/Desktop/LLM/data_training/data_output_clean"
    # pred_folder = "/home/luciacev/Desktop/LLM/Qwen2.5-1.5B-instruct/predict_Qwen1.5B_all"
    pred_folder = "/home/luciacev/Desktop/LLM/Mistral-7B-instruct/predict_Mistral7B_eval"

    
    if not os.path.exists(ref_folder):
        print(f"❌ Reference folder not found: {ref_folder}")
        return
    
    if not os.path.exists(pred_folder):
        print(f"❌ Prediction folder not found: {pred_folder}")
        return
    
    print(f"\n📁 References: {ref_folder}")
    print(f"📁 Predictions: {pred_folder}\n")
    
    # ===== MODE 1: With unknowns =====
    print("="*120)
    print("MODE 1️⃣  : WITH UNKNOWNS (all values included)")
    print("="*120)
    
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
    
    print(f"\n📊 OVERALL SEMANTIC SIMILARITY (with unknowns):")
    print(f"   Mean: {overall_mean_normal:.4f}")
    print(f"   Std Dev: {overall_std_normal:.4f}")
    print(f"   Total comparisons: {len(all_scores_normal)}")
    
    # ===== MODE 2: Without unknowns =====
    print("\n" + "="*120)
    print("MODE 2️⃣  : WITHOUT UNKNOWNS (filtered)")
    print("="*120)
    
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
    print("\n" + "="*120)
    print("📊 COMPARISON: With vs Without Unknowns")
    print("="*120)
    print(f"\n{'Metric':<30} {'With Unknowns':<20} {'Without Unknowns':<20}")
    print("-"*70)
    print(f"{'Mean':<30} {overall_mean_normal:.4f}{' '*15} {overall_mean_filtered:.4f}")
    print(f"{'Std Dev':<30} {overall_std_normal:.4f}{' '*15} {overall_std_filtered:.4f}")
    print(f"{'Total comparisons':<30} {len(all_scores_normal):<20} {len(all_scores_filtered):<20}")
    
    # ===== PER-CRITERION COMPARISON =====
    print("\n" + "="*120)
    print("📋 PER-CRITERION SEMANTIC SIMILARITY SCORES")
    print("="*120)
    
    print(f"\n{'Criterion':<40} {'With UK':<15} {'No UK':<15}")
    print("-"*70)
    
    # Sort all criteria by normal score (best to worst)
    criterion_diffs = {}
    for criterion in CRITERIA:
        normal = criterion_stats_normal[criterion]['mean']
        filtered = criterion_stats_filtered[criterion]['mean']
        criterion_diffs[criterion] = {
            'normal': normal,
            'filtered': filtered
        }
    
    # Sort from best to worst
    sorted_criteria = sorted(criterion_diffs.items(), key=lambda x: x[1]['normal'], reverse=True)
    
    print("\n📊 ALL CRITERIA (ranked from best to worst):")
    for criterion, stats in sorted_criteria:
        print(f"{criterion:<40} {stats['normal']:.4f}          {stats['filtered']:.4f}")
    



if __name__ == "__main__":
    main()
