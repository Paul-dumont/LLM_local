import argparse
import os
import json
from collections import defaultdict
from pathlib import Path

# === Configuration (top-level) ===
# Set the model short name and derive the model folder automatically
# Convention: model_folder = f"{model}-instruct"
model = 'Qwen7B'
model_folder = f"{model}-instruct"
REPO_ROOT = Path(__file__).resolve().parents[1]

# Define the 46 criteria in the exact display order requested by the user
CRITERIA = [
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
    "fibromyalgia_present"
]

def parse_summary_file(file_path):
    """Parse a summary file into a dictionary of key-value pairs."""
    data = {}
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '=' in line:
                key, value = line.split('=', 1)
            elif ':' in line:
                key, value = line.split(':', 1)
            else:
                continue
            key = key.strip()
            value = value.strip()
            if key in CRITERIA:
                data[key] = value
    return data

def calculate_metrics(manual_data, llm_data, exclude_unknown=False):
    """Return raw TP, FP, FN counts per field (standard approach).
    
    Args:
        exclude_unknown: If True, only count cases where reference is NOT 'unknown'
    
    Standard approach:
    - TP: manual_value == llm_value (both match)
    - FP: llm_value exists but manual_value is None/missing
    - FN: manual_value exists but llm_value is None/missing or they don't match
    """
    metrics = {criterion: {'TP': 0, 'FP': 0, 'FN': 0} for criterion in CRITERIA}
    
    for criterion in CRITERIA:
        manual_value = manual_data.get(criterion, None)
        llm_value = llm_data.get(criterion, None)
        
        # Skip this criterion if excluding unknown and manual is unknown
        if exclude_unknown and manual_value == "unknown":
            continue
        
        if manual_value is not None and llm_value is not None:
            if manual_value == llm_value:
                metrics[criterion]['TP'] += 1
            else:
                # Mismatch: count as FN (we failed to extract the correct value)
                metrics[criterion]['FN'] += 1
        elif manual_value is not None and llm_value is None:
            # Manual exists but LLM didn't extract it
            metrics[criterion]['FN'] += 1
        elif manual_value is None and llm_value is not None:
            # LLM extracted something but there's no ground truth
            metrics[criterion]['FP'] += 1
    
    return metrics

def evaluate_folders(manual_folder, llm_folder, exclude_unknown=False):
    """Aggregate TP, FP, FN over all patients and compute final metrics.
    
    Args:
        exclude_unknown: If True, only count cases where reference is NOT 'unknown'
    """
    all_results = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
    presence_counts = defaultdict(lambda: {'present': 0, 'total': 0})  # Track presence of criteria
    
    llm_files = [f for f in os.listdir(llm_folder) if f.endswith('_pred.txt')]

    for llm_file in llm_files:
        patient_id = llm_file.replace('_pred.txt', '')
        # Support multiple reference naming conventions (e.g. B001_summary.txt or B001.txt)
        manual_candidates = [
            f"{patient_id}_summary.txt",
            f"{patient_id}.txt",
        ]

        manual_path = None
        for candidate in manual_candidates:
            candidate_path = os.path.join(manual_folder, candidate)
            if os.path.exists(candidate_path):
                manual_path = candidate_path
                break

        llm_path = os.path.join(llm_folder, llm_file)
        
        if manual_path is not None:
            manual_data = parse_summary_file(manual_path)
            llm_data = parse_summary_file(llm_path)
            
            patient_metrics = calculate_metrics(manual_data, llm_data, exclude_unknown=exclude_unknown)
            
            # Count presence of criteria in reference data
            for criterion in CRITERIA:
                manual_value = manual_data.get(criterion, None)
                presence_counts[criterion]['total'] += 1
                if manual_value is not None and manual_value != 'unknown':
                    presence_counts[criterion]['present'] += 1
            
            for criterion, counts in patient_metrics.items():
                all_results[criterion]['TP'] += counts['TP']
                all_results[criterion]['FP'] += counts['FP']
                all_results[criterion]['FN'] += counts['FN']
    
    # Compute metrics per field from accumulated TP, FP, FN
    avg_results = {}
    for criterion in CRITERIA:
        TP = all_results[criterion]['TP']
        FP = all_results[criterion]['FP']
        FN = all_results[criterion]['FN']
        
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        Accuracy = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0

        # Get presence count
        presence = presence_counts[criterion]['present']
        total = presence_counts[criterion]['total']
        
        avg_results[criterion] = {
            'Presence': f"{presence}/{total}",
            'Precision': round(Precision, 4),
            'Recall': round(Recall, 4),
            'Accuracy': round(Accuracy, 4),
            'F1': round(F1, 4)
        }
    
    return avg_results, all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute per-criterion and global F1 between a ground-truth folder and a predictions folder.")
    parser.add_argument("--manual", type=str, default=None, help="Ground-truth folder (defaults to output_500_raw)")
    parser.add_argument("--llm", type=str, default=None, help="Predictions folder (defaults to Qwen7B-instruct/pred_500_raw)")
    parser.add_argument("--label", type=str, default=model, help="Label used in the printed header / json output")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to dump the global macro/micro metrics as JSON")
    args = parser.parse_args()

    # Input/output folders to evaluate
    manual_folder = args.manual or str(REPO_ROOT / "data_training/500_1600_raw_harmo/output_500_raw")
    llm_folder = args.llm or str(REPO_ROOT / "Qwen7B-instruct/pred_500_raw")

    if not os.path.exists(manual_folder) or not os.path.exists(llm_folder):
        print("Error: Folders not found.")
    else:
        # Calculate both versions
        print("Computing F1 scores...")
        # Version 1: WITH unknown (original)
        avg_metrics, _ = evaluate_folders(manual_folder, llm_folder, exclude_unknown=False)

        # Version 2: WITHOUT unknown (real data only)

        avg_metrics_no_unknown, raw_metrics_no_unknown = evaluate_folders(manual_folder, llm_folder, exclude_unknown=True)
        
        # Note: per user request we do NOT save metrics to disk anymore.
        # The metrics are computed and printed below but not written to files.

        
        # Calculate and print F1 statistics for BOTH
        # Filter out criteria with presence 0/100
        
        def extract_presence_count(presence_str):
            """Extract the numerator from 'X/Y' format"""
            try:
                return int(presence_str.split('/')[0])
            except:
                return 0
        
        # Filter criteria with at least 1 presence
        filtered_metrics = {}
        filtered_metrics_no_unk = {}
        
        for criterion in CRITERIA:
            presence_str = avg_metrics.get(criterion, {}).get('Presence', '0/0')
            presence_count = extract_presence_count(presence_str)
            
            if presence_count > 0:  # Only include if present at least once
                filtered_metrics[criterion] = avg_metrics[criterion]
                filtered_metrics_no_unk[criterion] = avg_metrics_no_unknown[criterion]
        
        f1_scores = [v['F1'] for v in filtered_metrics.values()]
        f1_scores_no_unk = [v['F1'] for v in filtered_metrics_no_unk.values()]
        precision_scores = [v['Precision'] for v in filtered_metrics.values()]
        precision_scores_no_unk = [v['Precision'] for v in filtered_metrics_no_unk.values()]
        recall_scores = [v['Recall'] for v in filtered_metrics.values()]
        recall_scores_no_unk = [v['Recall'] for v in filtered_metrics_no_unk.values()]

        # Micro F1 (excluding unknown): pool TP/FP/FN across all filtered criteria, then compute one P/R/F1
        TP_micro = sum(raw_metrics_no_unknown[c]['TP'] for c in filtered_metrics_no_unk)
        FP_micro = sum(raw_metrics_no_unknown[c]['FP'] for c in filtered_metrics_no_unk)
        FN_micro = sum(raw_metrics_no_unknown[c]['FN'] for c in filtered_metrics_no_unk)
        precision_micro_no_unk = TP_micro / (TP_micro + FP_micro) if (TP_micro + FP_micro) > 0 else 0
        recall_micro_no_unk = TP_micro / (TP_micro + FN_micro) if (TP_micro + FN_micro) > 0 else 0
        f1_micro_no_unk = (
            2 * precision_micro_no_unk * recall_micro_no_unk / (precision_micro_no_unk + recall_micro_no_unk)
            if (precision_micro_no_unk + recall_micro_no_unk) > 0 else 0
        )
        
        if f1_scores and f1_scores_no_unk:
            avg_f1 = sum(f1_scores) / len(f1_scores)
            avg_f1_no_unk = sum(f1_scores_no_unk) / len(f1_scores_no_unk)
            avg_precision = sum(precision_scores) / len(precision_scores)
            avg_precision_no_unk = sum(precision_scores_no_unk) / len(precision_scores_no_unk)
            avg_recall = sum(recall_scores) / len(recall_scores)
            avg_recall_no_unk = sum(recall_scores_no_unk) / len(recall_scores_no_unk)
            
            std_f1 = (sum((x - avg_f1)**2 for x in f1_scores) / len(f1_scores))**0.5
            std_f1_no_unk = (sum((x - avg_f1_no_unk)**2 for x in f1_scores_no_unk) / len(f1_scores_no_unk))**0.5
            std_precision = (sum((x - avg_precision)**2 for x in precision_scores) / len(precision_scores))**0.5
            std_precision_no_unk = (sum((x - avg_precision_no_unk)**2 for x in precision_scores_no_unk) / len(precision_scores_no_unk))**0.5
            std_recall = (sum((x - avg_recall)**2 for x in recall_scores) / len(recall_scores))**0.5
            std_recall_no_unk = (sum((x - avg_recall_no_unk)**2 for x in recall_scores_no_unk) / len(recall_scores_no_unk))**0.5
            
            print(f"\n{'Metric':<25} {'WITH unknown':<20} {'WITHOUT unknown':<20}")
            print("-" * 75)
            print(f"{'Average Precision':<25} {avg_precision:<20.4f} {avg_precision_no_unk:<20.4f}")
            print(f"{'Average Recall':<25} {avg_recall:<20.4f} {avg_recall_no_unk:<20.4f}")
            print(f"{'Average F1 (filtered)':<25} {avg_f1:<20.4f} {avg_f1_no_unk:<20.4f}")
            print(f"{'Criteria count':<25} {len(f1_scores):<20} {len(f1_scores_no_unk):<20}")

            print(f"\n{'--- F1 excluding unknown (macro vs micro) ---'}")
            print(f"{'Macro F1 (avg per criterion)':<32} {avg_f1_no_unk:<10.4f}")
            print(f"{'Micro F1 (pooled TP/FP/FN)':<32} {f1_micro_no_unk:<10.4f}")
            print(f"{'Micro Precision':<32} {precision_micro_no_unk:<10.4f}")
            print(f"{'Micro Recall':<32} {recall_micro_no_unk:<10.4f}")

            # Detailed per-criterion comparison (only for criteria with presence > 0)
            print(f"\n{'Criterion':<40} {'Presence':<12} {'WITH unknown':<18} {'WITHOUT unknown':<18}")
            print("-" * 100)
            
            # Print in the user-requested feature order (CRITERIA list)
            for criterion in CRITERIA:
                if criterion in filtered_metrics:
                    metrics_with = filtered_metrics[criterion]
                    presence = metrics_with.get('Presence', 'N/A')
                    f1_with = metrics_with.get('F1', 0.0)
                    f1_without = filtered_metrics_no_unk[criterion].get('F1', 0.0)
                    print(f"{criterion:<40} {presence:<12} {f1_with:<18.4f} {f1_without:<18.4f}")
            
            print("=" * 100 + "\n")

            if args.json_out:
                summary = {
                    "label": args.label,
                    "manual_folder": manual_folder,
                    "llm_folder": llm_folder,
                    "n_criteria": len(f1_scores_no_unk),
                    "macro_f1_no_unknown": round(avg_f1_no_unk, 4),
                    "macro_precision_no_unknown": round(avg_precision_no_unk, 4),
                    "macro_recall_no_unknown": round(avg_recall_no_unk, 4),
                    "micro_f1_no_unknown": round(f1_micro_no_unk, 4),
                    "micro_precision_no_unknown": round(precision_micro_no_unk, 4),
                    "micro_recall_no_unknown": round(recall_micro_no_unk, 4),
                    "macro_f1_with_unknown": round(avg_f1, 4),
                    "per_criterion_f1_no_unknown": {c: v["F1"] for c, v in filtered_metrics_no_unk.items()},
                    "per_criterion_presence": {c: v["Presence"] for c, v in filtered_metrics.items()},
                }
                Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
                Path(args.json_out).write_text(json.dumps(summary, indent=2))
                print(f"Saved summary metrics -> {args.json_out}")
        else:
            print("No F1 scores to analyze.")