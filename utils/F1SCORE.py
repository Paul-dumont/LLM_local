import os
import json
from collections import defaultdict
from pathlib import Path

# === Configuration (top-level) ===
# Set the model short name and derive the model folder automatically
# Convention: model_folder = f"{model}-instruct"
model = 'Mistral7B'
model_folder = f"{model}-instruct"
REPO_ROOT = Path(__file__).resolve().parents[1]

# Define the 46 criteria in the exact display order requested by the user
CRITERIA = [
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
    """Return raw TP, FP, FN counts per field.
    
    Args:
        exclude_unknown: If True, only count cases where reference is NOT 'unknown'
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
                metrics[criterion]['FP'] += 1
                metrics[criterion]['FN'] += 1
        elif manual_value is not None:
            metrics[criterion]['FN'] += 1
        elif llm_value is not None:
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
        manual_file = f"{patient_id}_summary.txt"
        
        manual_path = os.path.join(manual_folder, manual_file)
        llm_path = os.path.join(llm_folder, llm_file)
        
        if os.path.exists(manual_path):
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
    
    return avg_results

if __name__ == "__main__":
    # Use ONLY the harmonized data_output as the manual/reference folder
    manual_folder = str(REPO_ROOT / 'data_predicition' / 'data_output_harmonized')

    # llm_folder is derived from the top-level `model` and `model_folder` variables
    # Use the harmonized predictions folder (suffix `_harmonized`) so we compare
    # predictions that were harmonized to the reference harmonized outputs.
    llm_folder = str(REPO_ROOT / model_folder / f"predict_{model}_eval_harmonized")
    
    if not os.path.exists(manual_folder) or not os.path.exists(llm_folder):
        print("Error: Folders not found.")
    else:
        # Calculate both versions
        print("Computing F1 scores...")
        # Version 1: WITH unknown (original)
        avg_metrics = evaluate_folders(manual_folder, llm_folder, exclude_unknown=False)
        
        # Version 2: WITHOUT unknown (real data only)

        avg_metrics_no_unknown = evaluate_folders(manual_folder, llm_folder, exclude_unknown=True)
        
        # Note: per user request we do NOT save metrics to disk anymore.
        # The metrics are computed and printed below but not written to files.

        
        # Calculate and print F1 statistics for BOTH

        
        f1_scores = [v['F1'] for v in avg_metrics.values()]
        f1_scores_no_unk = [v['F1'] for v in avg_metrics_no_unknown.values()]
        
        if f1_scores and f1_scores_no_unk:
            avg_f1 = sum(f1_scores) / len(f1_scores)
            avg_f1_no_unk = sum(f1_scores_no_unk) / len(f1_scores_no_unk)
            
            sorted_f1 = sorted(f1_scores)
            median_f1 = (sorted_f1[len(f1_scores)//2 - 1] + sorted_f1[len(f1_scores)//2]) / 2 if len(f1_scores) % 2 == 0 else sorted_f1[len(f1_scores)//2]
            
            sorted_f1_no_unk = sorted(f1_scores_no_unk)
            median_f1_no_unk = (sorted_f1_no_unk[len(f1_scores_no_unk)//2 - 1] + sorted_f1_no_unk[len(f1_scores_no_unk)//2]) / 2 if len(f1_scores_no_unk) % 2 == 0 else sorted_f1_no_unk[len(f1_scores_no_unk)//2]
            
            min_f1 = min(f1_scores)
            max_f1 = max(f1_scores)
            min_f1_no_unk = min(f1_scores_no_unk)
            max_f1_no_unk = max(f1_scores_no_unk)
            
            std_f1 = (sum((x - avg_f1)**2 for x in f1_scores) / len(f1_scores))**0.5
            std_f1_no_unk = (sum((x - avg_f1_no_unk)**2 for x in f1_scores_no_unk) / len(f1_scores_no_unk))**0.5
            
            print(f"\n{'Metric':<20} {'WITH unknown':<20} {'WITHOUT unknown':<20}")
            print("-" * 75)
            print(f"{'Average F1':<20} {avg_f1:<20.4f} {avg_f1_no_unk:<20.4f} ")

            
            
            # Detailed per-criterion comparison
            
            print(f"\n{'Criterion':<40} {'Presence':<12} {'WITH unknown':<18} {'WITHOUT unknown':<18}")
            print("-" * 100)
            
            # Print in the user-requested feature order (CRITERIA list)
            ordered_criteria = [(c, avg_metrics.get(c, {'Presence': '0/0', 'F1': 0.0})) for c in CRITERIA]

            for criterion, metrics_with in ordered_criteria:
                presence = metrics_with.get('Presence', 'N/A')
                f1_with = metrics_with.get('F1', 0.0)
                f1_without = avg_metrics_no_unknown.get(criterion, {}).get('F1', 0.0)

                print(f"{criterion:<40} {presence:<12} {f1_with:<18.4f} {f1_without:<18.4f}")
            
            print("=" * 100 + "\n")
        else:
            print("No F1 scores to analyze.")