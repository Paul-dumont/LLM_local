import os
import json
from collections import defaultdict

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
    manual_folder = "/home/luciacev/Desktop/LLM/Data_output_13"
    llm_folder = "/home/luciacev/Desktop/LLM/training/predictions_12_harmonized"
    
    if not os.path.exists(manual_folder) or not os.path.exists(llm_folder):
        print("Error: Folders not found.")
    else:
        # Calculate both versions
        print("=" * 80)
        print("CALCULATING F1 SCORES - WITH AND WITHOUT UNKNOWN VALUES")
        print("=" * 80)
        
        # Version 1: WITH unknown (original)
        print("\n[1/2] Computing F1 scores WITH unknown values...")
        avg_metrics = evaluate_folders(manual_folder, llm_folder, exclude_unknown=False)
        
        # Version 2: WITHOUT unknown (real data only)
        print("[2/2] Computing F1 scores WITHOUT unknown values...")
        avg_metrics_no_unknown = evaluate_folders(manual_folder, llm_folder, exclude_unknown=True)
        
        os.makedirs("metrics", exist_ok=True)
        
        # Save both versions
        with open("metrics/F1score.json", "w") as f:
            json.dump(avg_metrics, f, indent=4)
        
        with open("metrics/F1score_no_unknown.json", "w") as f:
            json.dump(avg_metrics_no_unknown, f, indent=4)
        
        print("\n✅ Metrics saved:")
        print("   - metrics/F1score.json (WITH unknown)")
        print("   - metrics/F1score_no_unknown.json (WITHOUT unknown)")
        
        # Calculate and print F1 statistics for BOTH
        print("\n" + "=" * 80)
        print("F1 COMPARISON: WITH UNKNOWN vs WITHOUT UNKNOWN")
        print("=" * 80)
        
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
            
            print(f"\n{'Metric':<20} {'WITH unknown':<20} {'WITHOUT unknown':<20} {'Difference':<15}")
            print("-" * 75)
            print(f"{'Average F1':<20} {avg_f1:<20.4f} {avg_f1_no_unk:<20.4f} {avg_f1_no_unk - avg_f1:+.4f}")
            print(f"{'Median F1':<20} {median_f1:<20.4f} {median_f1_no_unk:<20.4f} {median_f1_no_unk - median_f1:+.4f}")
            print(f"{'Min F1':<20} {min_f1:<20.4f} {min_f1_no_unk:<20.4f} {min_f1_no_unk - min_f1:+.4f}")
            print(f"{'Max F1':<20} {max_f1:<20.4f} {max_f1_no_unk:<20.4f} {max_f1_no_unk - max_f1:+.4f}")
            print(f"{'Std Dev F1':<20} {std_f1:<20.4f} {std_f1_no_unk:<20.4f} {std_f1_no_unk - std_f1:+.4f}")
            
            print("\n" + "=" * 80)
            print("INTERPRETATION:")
            print("=" * 80)
            print(f"✓ Real F1 performance is likely: {avg_f1_no_unk:.4f}")
            print(f"✗ Inflation from 'unknown' predictions: {avg_f1 - avg_f1_no_unk:+.4f}")
            print("=" * 80)
            
            # Detailed per-criterion comparison
            print("\n" + "=" * 100)
            print("DETAILED F1 SCORES PER CRITERION - WITH vs WITHOUT UNKNOWN")
            print("=" * 100)
            
            print(f"\n{'Criterion':<40} {'Presence':<12} {'WITH unknown':<18} {'WITHOUT unknown':<18}")
            print("-" * 100)
            
            sorted_criteria = sorted(avg_metrics.items(), key=lambda x: x[1]['F1'], reverse=True)
            
            for criterion, metrics_with in sorted_criteria:
                presence = metrics_with.get('Presence', 'N/A')
                f1_with = metrics_with['F1']
                f1_without = avg_metrics_no_unknown[criterion]['F1']
                
                print(f"{criterion:<40} {presence:<12} {f1_with:<18.4f} {f1_without:<18.4f}")
            
            print("=" * 100 + "\n")
        else:
            print("No F1 scores to analyze.")