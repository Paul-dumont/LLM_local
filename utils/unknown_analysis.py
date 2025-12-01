#!/usr/bin/env python3
"""
Unknown Values Analysis
For each criterion in data_output_clean, count how many "unknown" values
and display as count/total and percentage
"""

import os
from pathlib import Path
from collections import defaultdict
import numpy as np

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


def extract_criterion_from_file(filepath):
    """Extract criterion values from a file"""
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


def analyze_unknowns(data_folder):
    """Analyze unknown values for each criterion across all patients"""
    
    criterion_stats = defaultdict(lambda: {"unknown_count": 0, "total": 0})
    
    data_path = Path(data_folder)
    files = sorted(data_path.glob("B*_summary.txt"))
    
    print(f"\n📊 Found {len(files)} patient files")
    print(f"📁 Folder: {data_folder}\n")
    
    # Process each file
    for file in files:
        values = extract_criterion_from_file(str(file))
        
        for criterion in CRITERIA:
            value = values.get(criterion, "").strip()
            
            criterion_stats[criterion]["total"] += 1
            if is_unknown(value):
                criterion_stats[criterion]["unknown_count"] += 1
    
    return criterion_stats


def main():
    print("\n" + "="*100)
    print("📋 UNKNOWN VALUES ANALYSIS")
    print("="*100)
    
    data_folder = "/home/luciacev/Desktop/LLM/data_training/data_output_clean"
    
    if not os.path.exists(data_folder):
        print(f"❌ Folder not found: {data_folder}")
        return
    
    criterion_stats = analyze_unknowns(data_folder)
    
    # Calculate percentages and sort by available percentage (highest first)
    stats_with_percent = []
    for criterion in CRITERIA:
        stats = criterion_stats[criterion]
        unknown_count = stats["unknown_count"]
        total = stats["total"]
        available_count = total - unknown_count
        percent = (available_count / total * 100) if total > 0 else 0
        
        stats_with_percent.append({
            "criterion": criterion,
            "available_count": available_count,
            "unknown_count": unknown_count,
            "total": total,
            "percent": percent
        })
    
    # Sort by percentage (descending)
    stats_with_percent.sort(key=lambda x: x["percent"], reverse=True)
    
    # ===== DISPLAY RESULTS =====
    print(f"\n{'Criterion':<40} {'Available Count':<20} {'Percentage':<15}")
    print("-"*75)
    
    for stat in stats_with_percent:
        criterion = stat["criterion"]
        available = stat["available_count"]
        total = stat["total"]
        percent = stat["percent"]
        
        print(f"{criterion:<40} {available}/{total:<17} {percent:>6.2f}%")
    
    # ===== SUMMARY STATISTICS =====
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    
    all_unknown_counts = [s["unknown_count"] for s in stats_with_percent]
    all_available_counts = [s["available_count"] for s in stats_with_percent]
    all_totals = [s["total"] for s in stats_with_percent]
    all_percents = [s["percent"] for s in stats_with_percent]
    
    total_unknown = sum(all_unknown_counts)
    total_available = sum(all_available_counts)
    total_values = sum(all_totals)
    overall_percent = (total_available / total_values * 100) if total_values > 0 else 0
    
    print(f"\nOverall Statistics:")
    print(f"   Total available values: {total_available}/{total_values} ({overall_percent:.2f}%)")
    print(f"   Mean available % per criterion: {np.mean(all_percents):.2f}%")
    print(f"   Median available % per criterion: {np.median(all_percents):.2f}%")
    print(f"   Max available % per criterion: {np.max(all_percents):.2f}%")
    print(f"   Min available % per criterion: {np.min(all_percents):.2f}%")
    
    print("\n" + "="*100)
    print("CRITERIA BY DATA QUALITY")
    print("="*100)
    
    excellent = [s for s in stats_with_percent if s["percent"] == 100]
    good = [s for s in stats_with_percent if 80 <= s["percent"] < 100]
    medium = [s for s in stats_with_percent if 50 <= s["percent"] < 80]
    poor = [s for s in stats_with_percent if s["percent"] < 50]
    
    print(f"\nEXCELLENT (100% available): {len(excellent)} criteria")
    for s in excellent:
        print(f"   - {s['criterion']}")
    
    print(f"\nGOOD (80-100% available): {len(good)} criteria")
    for s in good:
        print(f"   - {s['criterion']:<40} {s['percent']:>6.2f}%")
    
    print(f"\nMEDIUM (50-80% available): {len(medium)} criteria")
    for s in medium:
        print(f"   - {s['criterion']:<40} {s['percent']:>6.2f}%")
    
    print(f"\nPOOR (<50% available): {len(poor)} criteria")
    for s in poor:
        print(f"   - {s['criterion']:<40} {s['percent']:>6.2f}%")
    
    # ===== EXPORT JSON =====
    import json
    output_file = Path(__file__).parent.parent / "unknown_analysis.json"
    
    export_data = {
        "summary": {
            "total_available": total_available,
            "total_values": total_values,
            "overall_percentage": round(overall_percent, 2),
            "mean_available_percent": round(np.mean(all_percents), 2),
            "median_available_percent": round(np.median(all_percents), 2)
        },
        "by_criterion": [
            {
                "criterion": s["criterion"],
                "available_count": s["available_count"],
                "unknown_count": s["unknown_count"],
                "total": s["total"],
                "percentage": round(s["percent"], 2)
            }
            for s in stats_with_percent
        ],
        "quality_categories": {
            "excellent": [s["criterion"] for s in excellent],
            "good": [s["criterion"] for s in good],
            "medium": [s["criterion"] for s in medium],
            "poor": [s["criterion"] for s in poor]
        }
    }
    
    output_file.write_text(json.dumps(export_data, indent=2), encoding='utf-8')
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
