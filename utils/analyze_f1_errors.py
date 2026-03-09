import os
import json
from collections import defaultdict
from pathlib import Path

# === Configuration ===
model = 'Qwen7B'
model_folder = f"{model}-instruct"
REPO_ROOT = Path(__file__).resolve().parents[1]

CRITERIA = [
    "patient_id", "patient_age", "headache_intensity", "tmj_pain_rating",
    "disc_displacement", "joint_arthritis_location", "jaw_function_score",
    "maximum_opening", "diet_score", "disability_rating", "tinnitus_present",
    "vertigo_present", "joint_pain_areas", "earache_present",
    "pain_aggravating_factors", "average_daily_pain_intensity",
    "airway_obstruction_present", "pain_onset_date", "appliance_history",
    "current_medications", "headache_location", "muscle_pain_location",
    "muscle_symptoms_present", "muscle_pain_score", "hearing_loss_present",
    "jaw_clicking", "headache_frequency", "sleep_disorder_type",
    "maximum_opening_without_pain", "neck_pain_present", "current_appliance",
    "onset_triggers", "physical_therapy_status", "adverse_reactions",
    "jaw_crepitus", "jaw_locking", "pain_relieving_factors", "back_pain_present",
    "sleep_apnea_diagnosed", "autoimmune_condition", "migraine_history",
    "previous_medications", "pain_frequency", "depression_present",
    "pain_duration", "fibromyalgia_present"
]

def parse_summary_file(file_path):
    """Parse a summary file into a dictionary."""
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

def analyze_errors_detailed(manual_folder, llm_folder):
    """Analyze errors in detail for each criterion."""
    
    error_analysis = {criterion: {
        'errors': [],
        'TP': 0, 'FP': 0, 'FN': 0,
        'total_samples': 0,
        'presence': 0,
        'error_types': defaultdict(int)  # Track error patterns
    } for criterion in CRITERIA}
    
    llm_files = [f for f in os.listdir(llm_folder) if f.endswith('_pred.txt')]
    
    for llm_file in llm_files:
        patient_id = llm_file.replace('_pred.txt', '')
        manual_file = f"{patient_id}_summary.txt"
        
        manual_path = os.path.join(manual_folder, manual_file)
        llm_path = os.path.join(llm_folder, llm_file)
        
        if os.path.exists(manual_path):
            manual_data = parse_summary_file(manual_path)
            llm_data = parse_summary_file(llm_path)
            
            for criterion in CRITERIA:
                manual_value = manual_data.get(criterion, None)
                llm_value = llm_data.get(criterion, None)
                
                error_analysis[criterion]['total_samples'] += 1
                
                if manual_value is not None and manual_value != 'unknown':
                    error_analysis[criterion]['presence'] += 1
                
                # TP: correct match
                if manual_value is not None and llm_value is not None:
                    if manual_value == llm_value:
                        error_analysis[criterion]['TP'] += 1
                    else:
                        # FP + FN: mismatch
                        error_analysis[criterion]['FP'] += 1
                        error_analysis[criterion]['FN'] += 1
                        
                        # Categorize error type
                        if manual_value == 'unknown':
                            error_type = 'hallucination'
                        elif llm_value == 'unknown':
                            error_type = 'missed_field'
                        else:
                            error_type = 'wrong_value'
                        
                        error_analysis[criterion]['error_types'][error_type] += 1
                        
                        # Store error details
                        error_analysis[criterion]['errors'].append({
                            'patient_id': patient_id,
                            'expected': manual_value,
                            'predicted': llm_value,
                            'type': error_type
                        })
                
                # FN: missing from LLM
                elif manual_value is not None and llm_value is None:
                    error_analysis[criterion]['FN'] += 1
                    error_analysis[criterion]['error_types']['missing'] += 1
                    error_analysis[criterion]['errors'].append({
                        'patient_id': patient_id,
                        'expected': manual_value,
                        'predicted': 'MISSING',
                        'type': 'missing'
                    })
                
                # FP: extra from LLM
                elif manual_value is None and llm_value is not None:
                    error_analysis[criterion]['FP'] += 1
                    error_analysis[criterion]['error_types']['extra'] += 1
                    error_analysis[criterion]['errors'].append({
                        'patient_id': patient_id,
                        'expected': 'NONE',
                        'predicted': llm_value,
                        'type': 'extra'
                    })
    
    return error_analysis

def generate_report(error_analysis):
    """Generate a human-readable report for the paper."""
    
    report = []
    report.append("=" * 120)
    report.append("ANALYSIS OF F1 SCORE ERRORS - DETAILED BREAKDOWN")
    report.append("=" * 120)
    report.append("")
    
    # Sort criteria by F1 score (worst first)
    sorted_criteria = []
    for criterion in CRITERIA:
        analysis = error_analysis[criterion]
        TP = analysis['TP']
        FP = analysis['FP']
        FN = analysis['FN']
        
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0
        
        sorted_criteria.append((criterion, F1, analysis))
    
    sorted_criteria.sort(key=lambda x: x[1])  # Sort by F1 score (ascending)
    
    # === SECTION 1: Problem Areas ===
    report.append("\n1. CRITERIA WITH LOWEST F1 SCORES (Most Problematic)")
    report.append("-" * 120)
    
    worst_10 = sorted_criteria[:10]
    for criterion, f1, analysis in worst_10:
        TP = analysis['TP']
        FP = analysis['FP']
        FN = analysis['FN']
        presence = analysis['presence']
        total = analysis['total_samples']
        
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        report.append(f"\n📌 {criterion.upper()}")
        report.append(f"   F1 Score: {f1:.4f} | Precision: {Precision:.4f} | Recall: {Recall:.4f}")
        report.append(f"   Data Availability: {presence}/{total} ({100*presence/total:.1f}%)")
        report.append(f"   TP: {TP}, FP: {FP}, FN: {FN}")
        
        # Error type breakdown
        error_types = analysis['error_types']
        if error_types:
            report.append(f"   Error Breakdown:")
            for error_type, count in sorted(error_types.items(), key=lambda x: -x[1]):
                report.append(f"      - {error_type}: {count} cases ({100*count/(TP+FP+FN):.1f}%)")
        
        # Show sample errors (max 3)
        errors = analysis['errors'][:3]
        if errors:
            report.append(f"   Sample Errors:")
            for err in errors:
                report.append(f"      • Patient {err['patient_id']}: Expected '{err['expected']}' → Got '{err['predicted']}' ({err['type']})")
    
    # === SECTION 2: Best Performers ===
    report.append("\n\n2. CRITERIA WITH HIGHEST F1 SCORES (Well-Handled)")
    report.append("-" * 120)
    
    best_10 = sorted_criteria[-10:]
    best_10.reverse()
    for criterion, f1, analysis in best_10:
        TP = analysis['TP']
        FP = analysis['FP']
        FN = analysis['FN']
        presence = analysis['presence']
        total = analysis['total_samples']
        
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        
        report.append(f"\n✓ {criterion}")
        report.append(f"   F1: {f1:.4f} | Precision: {Precision:.4f} | Recall: {Recall:.4f} | Availability: {presence}/{total}")
    
    # === SECTION 3: Error Patterns by Type ===
    report.append("\n\n3. GLOBAL ERROR PATTERNS")
    report.append("-" * 120)
    
    global_errors = defaultdict(int)
    total_errors = 0
    for criterion, f1, analysis in sorted_criteria:
        for error_type, count in analysis['error_types'].items():
            global_errors[error_type] += count
            total_errors += count
    
    report.append("\nError Type Distribution (across all criteria):")
    for error_type in ['hallucination', 'missed_field', 'wrong_value', 'missing', 'extra']:
        if error_type in global_errors:
            count = global_errors[error_type]
            pct = 100 * count / total_errors if total_errors > 0 else 0
            report.append(f"   • {error_type}: {count} ({pct:.1f}%)")
    
    # === SECTION 4: Data Sparsity Issues ===
    report.append("\n\n4. DATA SPARSITY ANALYSIS (Low Presence Criteria)")
    report.append("-" * 120)
    
    sparse_criteria = [(c, a) for c, f1, a in sorted_criteria if a['presence'] < 30]
    report.append(f"\nCriteria with <30% presence in training data ({len(sparse_criteria)} criteria):")
    for criterion, analysis in sorted(sparse_criteria, key=lambda x: x[1]['presence']):
        presence = analysis['presence']
        total = analysis['total_samples']
        f1 = 2 * (analysis['TP'] / (analysis['TP'] + analysis['FP'])) * (analysis['TP'] / (analysis['TP'] + analysis['FN'])) if (analysis['TP'] + analysis['FP']) > 0 and (analysis['TP'] + analysis['FN']) > 0 else 0
        report.append(f"   • {criterion}: {presence}/{total} ({100*presence/total:.1f}%) - F1: {f1:.4f}")
    
    # === SECTION 5: Recommendations ===
    report.append("\n\n5. RECOMMENDATIONS FOR PAPER")
    report.append("-" * 120)
    
    recommendations = [
        "1. Data Augmentation: Focus on sparse criteria (presence < 30%)",
        "2. Prompt Engineering: Improve handling of complex fields (joint_pain_areas, pain_aggravating_factors)",
        "3. Post-Processing: Implement consistency checks for multi-part fields",
        "4. Fine-tuning Strategy: Weight loss towards low-F1 criteria",
        "5. Validation: Add domain-specific validation rules"
    ]
    
    for rec in recommendations:
        report.append(f"   {rec}")
    
    report.append("\n" + "=" * 120)
    
    return "\n".join(report)

if __name__ == "__main__":
    manual_folder = str(REPO_ROOT / 'data_predicition' / 'data_output_harmonized')
    llm_folder = str(REPO_ROOT / model_folder / "save_data_before_reduce_features" / "predict_Qwen7B_eval_harmonized")
    
    if not os.path.exists(manual_folder) or not os.path.exists(llm_folder):
        print("Error: Folders not found.")
    else:
        print("Analyzing F1 score errors...")
        error_analysis = analyze_errors_detailed(manual_folder, llm_folder)
        
        # Generate and print report
        report = generate_report(error_analysis)
        print(report)
        
        # Save report to file
        output_file = str(REPO_ROOT / 'utils' / 'F1_ERROR_ANALYSIS.txt')
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\n✓ Report saved to: {output_file}")
        
        # Also save as JSON for further analysis
        json_output = str(REPO_ROOT / 'utils' / 'F1_ERROR_ANALYSIS.json')
        json_data = {}
        for criterion, analysis in error_analysis.items():
            TP = analysis['TP']
            FP = analysis['FP']
            FN = analysis['FN']
            Precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            Recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0
            
            json_data[criterion] = {
                'F1': round(F1, 4),
                'Precision': round(Precision, 4),
                'Recall': round(Recall, 4),
                'TP': TP,
                'FP': FP,
                'FN': FN,
                'presence': analysis['presence'],
                'total_samples': analysis['total_samples'],
                'error_types': dict(analysis['error_types']),
                'sample_errors': analysis['errors'][:5]  # Store top 5 errors
            }
        
        with open(json_output, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"✓ JSON data saved to: {json_output}")
