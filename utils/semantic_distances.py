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
from datetime import datetime
import pandas as pd

# Try to import semantic model
try:
    from sentence_transformers import SentenceTransformer, util
    print("✅ Loading Semantic Model: all-MiniLM-L6-v2")
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    SEMANTIC_AVAILABLE = True
except Exception as e:
    print(f"⚠️  Semantic model not available: {e}")
    SEMANTIC_AVAILABLE = False

# === PATH ===
REPO_ROOT = Path(__file__).resolve().parents[1]

input_dir = str(REPO_ROOT / "data_training/500_1600_raw_harmo/output_1600_harmo")
output_dir = str(REPO_ROOT / "Qwen7B-instruct/pred_1600_harmo")

# Configure reference and prediction folders
REF_FOLDER = input_dir
PRED_FOLDER = output_dir

CRITERIA = []

# Preferred semantic order (highest → lowest representation)
DEFINED_SEMANTIC_ORDER = [
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

SEMANTIC_ORDER = []

UNKNOWN_VARIATIONS = ["unknown", "unknow", "n/a", "na", "not available", "not provided", "none", ""]


def extract_all_criteria(pred_folder):
    """
    Dynamically extract all criteria from prediction files.
    Returns a list of criteria in the preferred semantic order, with any additional criteria at the end.
    """
    criteria_set = set()
    
    pred_path = Path(pred_folder)
    
    # Scan prediction files only
    for pred_file in pred_path.glob("B*.txt"):
        try:
            with open(pred_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        key = line.split(':', 1)[0].strip()
                        if key and key != "patient_id":
                            criteria_set.add(key)
        except Exception as e:
            print(f"⚠️  Error reading {pred_file}: {e}")
    
    # Sort by preferred semantic order
    ordered_criteria = []
    for criterion in DEFINED_SEMANTIC_ORDER:
        if criterion in criteria_set:
            ordered_criteria.append(criterion)
            criteria_set.remove(criterion)
    
    # Add any remaining criteria not in DEFINED_SEMANTIC_ORDER (sorted alphabetically)
    if criteria_set:
        ordered_criteria.extend(sorted(list(criteria_set)))
    
    return ordered_criteria


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


def extract_criterion_from_file(filepath, criteria_list=None):
    """Extract criterion values from a prediction/reference file"""
    if not os.path.exists(filepath):
        return {}
    
    if criteria_list is None:
        criteria_list = CRITERIA
    
    values = {}
    
    # Initialize all criteria with empty strings
    for criterion in criteria_list:
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
                # Store any key found in the file
                if key and key != "patient_id":
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
    pred_files = sorted(pred_path.glob("B*.txt"))
    ref_files = sorted(ref_path.glob("B*.txt"))
    
    print(f"📊 Evaluating {len(pred_files)} patients...\n")
    
    for pred_file in pred_files:
        # Handle both naming schemes: B001.txt and B001_pred.txt
        filename = pred_file.name.replace(".txt", "")
        patient_id = filename.replace("_pred", "")  # Remove _pred suffix if present
        ref_file = ref_path / f"{patient_id}.txt"
        
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
    global CRITERIA, SEMANTIC_ORDER
    
    # Continue even if semantic model not available (fallback to exact match)
    if not SEMANTIC_AVAILABLE:
        print("⚠️  Semantic model not available - using exact match fallback")
    
    # Use the reference folder configured at the top
    manual_folder = REF_FOLDER

    # llm_folder is derived from the top-level `model` and `model_folder` variables
    # Use PRED_FOLDER configured at the top
    llm_folder = PRED_FOLDER

    # Keep backward-compatible names used later in the script
    ref_folder = manual_folder
    pred_folder = llm_folder

    # Cache prediction files count for reporting/output exports
    pred_files = sorted(Path(pred_folder).glob("B*.txt"))
    total_patients_evaluated = len(pred_files)

    
    if not os.path.exists(ref_folder):
        print(f"❌ Reference folder not found: {ref_folder}")
        return
    
    if not os.path.exists(pred_folder):
        print(f"❌ Prediction folder not found: {pred_folder}")
        return
    
    print(f"\n📁 References: {ref_folder}")
    print(f"📁 Predictions: {pred_folder}\n")
    
    # Dynamically extract criteria from prediction files
    print("🔍 Extracting criteria from prediction files...")
    CRITERIA = extract_all_criteria(pred_folder)
    SEMANTIC_ORDER = CRITERIA  # Ordered by preferred semantic representation
    print(f"✅ Found {len(CRITERIA)} criteria\n")
    
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
    
    # ===== SAVE RESULTS =====
    # Create result folder
    result_dir = REPO_ROOT / "result"
    result_dir.mkdir(exist_ok=True)
    
    # Use a simple output filename derived from the prediction folder
    pred_folder_name = Path(pred_folder).name
    output_filename = f"semantic_{pred_folder_name}.xlsx"
    output_path = result_dir / output_filename
    
    # Create Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Compact summary table requested by the user
        df_summary = pd.DataFrame(
            {
                'With unknow': [
                    overall_mean_normal,
                    len(all_scores_normal),
                    ref_folder,
                    pred_folder,
                ],
                'Without unknow': [
                    overall_mean_filtered,
                    len(all_scores_filtered),
                    ref_folder,
                    pred_folder,
                ],
            },
            index=['Mean', 'nb', 'Ref folder', 'Pred folder']
        )
        df_summary.to_excel(writer, sheet_name='Summary')

        summary_ws = writer.sheets['Summary']
        summary_ws['B2'].number_format = '0.0000'
        summary_ws['C2'].number_format = '0.0000'
        summary_ws['B3'].number_format = '0'
        summary_ws['C3'].number_format = '0'
        
        # Sheet 2: Per-Criterion Statistics
        criteria_data = {
            'Criterion': [],
            'With unknow': [],
            'Without unknow': [],
            'Count': []
        }
        
        for criterion in CRITERIA:
            criteria_data['Criterion'].append(criterion)
            criteria_data['With unknow'].append(criterion_stats_normal[criterion]['mean'])
            criteria_data['Without unknow'].append(criterion_stats_filtered[criterion]['mean'])
            criteria_data['Count'].append(criterion_stats_filtered[criterion]['count'])
        
        df_criteria = pd.DataFrame(criteria_data)
        df_criteria.to_excel(writer, sheet_name='Per-Criterion', index=False)
    
    print(f"\n✅ Results saved to: {output_path}")
    



if __name__ == "__main__":
    main()
