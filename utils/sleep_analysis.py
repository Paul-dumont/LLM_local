#!/usr/bin/env python3
"""
value_analysis.py - Simple value analysis for sleep features
"""

from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
DATA_OUTPUT_CLEAN = BASE_DIR / "data_predicition" / "data_output_harmonized"

def parse_patient_file(filepath: Path) -> dict:
    """Parse a patient summary file"""
    features = {}
    try:
        text = filepath.read_text(encoding='utf-8')
        for line in text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                if key in ['sleep_disorder_type', 'sleep_apnea_diagnosed']:
                    features[key] = value
    except Exception as e:
        pass
    return features

def analyze():
    """Analyze and display value counts"""
    note_files = sorted(DATA_OUTPUT_CLEAN.glob("*.txt"))
    
    # Store value counts
    values = {
        'sleep_disorder_type': defaultdict(int),
        'sleep_apnea_diagnosed': defaultdict(int)
    }
    
    # Store patient IDs by sleep_apnea_diagnosed value
    apnea_ids = {
        'true': [],
        'false': []
    }
    
    # Store patients with sleep apnea in sleep_disorder_type
    sleep_apnea_disorder_ids = []
    
    for note_file in note_files:
        features = parse_patient_file(note_file)
        for key, value in features.items():
            values[key][value] += 1
        
        # Extract patient ID and sleep_apnea_diagnosed
        if 'sleep_apnea_diagnosed' in features:
            apnea_value = features['sleep_apnea_diagnosed'].lower()
            if apnea_value in apnea_ids:
                patient_id = note_file.stem.replace('_summary', '')
                apnea_ids[apnea_value].append(patient_id)
        
        # Check if sleep_disorder_type contains "sleep apnea"
        if 'sleep_disorder_type' in features:
            disorder_value = features['sleep_disorder_type'].lower()
            if 'sleep apnea' in disorder_value:
                patient_id = note_file.stem.replace('_summary', '')
                sleep_apnea_disorder_ids.append(patient_id)
    
    # Display results
    for feature_name in ['sleep_disorder_type', 'sleep_apnea_diagnosed']:
        print(f"\n{'='*80}")
        print(f"FEATURE: {feature_name}")
        print(f"{'='*80}")
        
        # Sort by count descending
        sorted_values = sorted(values[feature_name].items(), key=lambda x: x[1], reverse=True)
        
        for value, count in sorted_values:
            percentage = (count / len(note_files)) * 100
            print(f"{count:4d} ({percentage:5.1f}%) : {value}")
    
    # Display patient IDs for sleep_apnea_diagnosed
    print(f"\n{'='*80}")
    print(f"PATIENT IDs - sleep_apnea_diagnosed = TRUE ({len(apnea_ids['true'])}):")
    print(f"{'='*80}")
    print(", ".join(apnea_ids['true']))
    
    print(f"\n{'='*80}")
    print(f"PATIENT IDs - sleep_apnea_diagnosed = FALSE ({len(apnea_ids['false'])}):")
    print(f"{'='*80}")
    print(", ".join(apnea_ids['false']))
    
    # Display patients with "sleep apnea" in sleep_disorder_type
    print(f"\n{'='*80}")
    print(f"PATIENT IDs - sleep_disorder_type contains 'sleep apnea' ({len(sleep_apnea_disorder_ids)}):")
    print(f"{'='*80}")
    print(", ".join(sorted(sleep_apnea_disorder_ids)))
    
    # Find intersection: sleep_apnea_diagnosed = TRUE AND sleep_disorder_type contains "sleep apnea"
    intersection = set(apnea_ids['true']) & set(sleep_apnea_disorder_ids)
    intersection_sorted = sorted(list(intersection))
    
    print(f"\n{'='*80}")
    print(f"INTERSECTION - sleep_apnea_diagnosed = TRUE AND sleep_disorder_type contains 'sleep apnea' ({len(intersection_sorted)}):")
    print(f"{'='*80}")
    print(", ".join(intersection_sorted))

if __name__ == "__main__":
    analyze()
