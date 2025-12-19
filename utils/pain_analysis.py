#!/usr/bin/env python3
"""
pain_analysis.py - Analyze pain_duration and pain_frequency features
"""

from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
DATA_OUTPUT_HARMONIZED = BASE_DIR / "data_predicition" / "data_output_harmonized"

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
                if key in ['pain_duration', 'pain_frequency']:
                    features[key] = value
    except Exception as e:
        pass
    return features

def analyze():
    """Analyze and display value counts"""
    note_files = sorted(DATA_OUTPUT_HARMONIZED.glob("*.txt"))
    
    # Store value counts
    values = {
        'pain_duration': defaultdict(int),
        'pain_frequency': defaultdict(int)
    }
    
    # Store patient data
    pain_data = {}
    
    for note_file in note_files:
        features = parse_patient_file(note_file)
        patient_id = note_file.stem.replace('_summary', '')
        
        for key, value in features.items():
            values[key][value] += 1
        
        pain_data[patient_id] = features
    
    # Display results
    for feature_name in ['pain_duration', 'pain_frequency']:
        print(f"\n{'='*80}")
        print(f"FEATURE: {feature_name}")
        print(f"{'='*80}")
        
        # Sort by count descending
        sorted_values = sorted(values[feature_name].items(), key=lambda x: x[1], reverse=True)
        
        for value, count in sorted_values:
            percentage = (count / len(note_files)) * 100
            print(f"{count:4d} ({percentage:5.1f}%) : {value}")
    
    # Find intersections between morning/afternoon/evening
    time_keywords = ['morning', 'afternoon', 'evening']
    
    for keyword in time_keywords:
        print(f"\n{'='*80}")
        print(f"PATIENTS with '{keyword}' in BOTH pain_duration AND pain_frequency:")
        print(f"{'='*80}")
        
        both = []
        for patient_id, features in pain_data.items():
            duration = features.get('pain_duration', '').lower()
            frequency = features.get('pain_frequency', '').lower()
            
            if keyword in duration and keyword in frequency:
                both.append(patient_id)
        
        print(f"Count: {len(both)}")
        if both:
            print(", ".join(sorted(both)))
        print()
    
    # Find patients with morning in pain_duration
    print(f"{'='*80}")
    print(f"PATIENTS with 'morning' in pain_duration:")
    print(f"{'='*80}")
    morning_duration = [pid for pid, feat in pain_data.items() if 'morning' in feat.get('pain_duration', '').lower()]
    print(f"Count: {len(morning_duration)}")
    print(", ".join(sorted(morning_duration)))
    
    # Find patients with morning in pain_frequency
    print(f"\n{'='*80}")
    print(f"PATIENTS with 'morning' in pain_frequency:")
    print(f"{'='*80}")
    morning_frequency = [pid for pid, feat in pain_data.items() if 'morning' in feat.get('pain_frequency', '').lower()]
    print(f"Count: {len(morning_frequency)}")
    print(", ".join(sorted(morning_frequency)))
    
    # Find patients with afternoon in pain_duration
    print(f"\n{'='*80}")
    print(f"PATIENTS with 'afternoon' in pain_duration:")
    print(f"{'='*80}")
    afternoon_duration = [pid for pid, feat in pain_data.items() if 'afternoon' in feat.get('pain_duration', '').lower()]
    print(f"Count: {len(afternoon_duration)}")
    if afternoon_duration:
        print(", ".join(sorted(afternoon_duration)))
    
    # Find patients with afternoon in pain_frequency
    print(f"\n{'='*80}")
    print(f"PATIENTS with 'afternoon' in pain_frequency:")
    print(f"{'='*80}")
    afternoon_frequency = [pid for pid, feat in pain_data.items() if 'afternoon' in feat.get('pain_frequency', '').lower()]
    print(f"Count: {len(afternoon_frequency)}")
    if afternoon_frequency:
        print(", ".join(sorted(afternoon_frequency)))
    
    # Find patients with evening in pain_duration
    print(f"\n{'='*80}")
    print(f"PATIENTS with 'evening' in pain_duration:")
    print(f"{'='*80}")
    evening_duration = [pid for pid, feat in pain_data.items() if 'evening' in feat.get('pain_duration', '').lower()]
    print(f"Count: {len(evening_duration)}")
    if evening_duration:
        print(", ".join(sorted(evening_duration)))
    
    # Find patients with evening in pain_frequency
    print(f"\n{'='*80}")
    print(f"PATIENTS with 'evening' in pain_frequency:")
    print(f"{'='*80}")
    evening_frequency = [pid for pid, feat in pain_data.items() if 'evening' in feat.get('pain_frequency', '').lower()]
    print(f"Count: {len(evening_frequency)}")
    if evening_frequency:
        print(", ".join(sorted(evening_frequency)))

if __name__ == "__main__":
    analyze()
