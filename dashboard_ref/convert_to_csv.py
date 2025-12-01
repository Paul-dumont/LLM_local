import os
import csv

data_dir = '/home/luciacev/Desktop/LLM/Data_output_harmonized'
files = [f for f in os.listdir(data_dir) if f.endswith('_summary.txt')]
print(f"Processing {len(files)} files")

all_keys = ['patient_id', 'patient_age', 'maximum_opening_without_pain', 'maximum_opening', 'pain_relieving_factors', 'diet_score', 'pain_duration', 'disability_rating', 'sleep_apnea_diagnosed', 'jaw_function_score', 'tinnitus_present', 'vertigo_present', 'hearing_loss_present', 'back_pain_present', 'neck_pain_present', 'pain_aggravating_factors', 'physical_therapy_status', 'earache_present', 'onset_triggers', 'adverse_reactions', 'fibromyalgia_present', 'jaw_locking', 'airway_obstruction_present', 'depression_present', 'headache_frequency', 'pain_frequency', 'muscle_symptoms_present', 'jaw_clicking', 'previous_medications', 'muscle_pain_score', 'migraine_history', 'headache_location', 'current_appliance', 'average_daily_pain_intensity', 'headache_intensity', 'autoimmune_condition', 'pain_onset_date', 'jaw_crepitus', 'tmj_pain_rating', 'sleep_disorder_type', 'current_medications', 'disc_displacement', 'muscle_pain_location', 'joint_arthritis_location', 'appliance_history', 'joint_pain_areas']

data = []

for file in files:
    with open(os.path.join(data_dir, file), 'r') as f:
        content = f.read()
        lines = content.strip().split('\n')
        patient_data = {key: '' for key in all_keys}  # default empty
        for line in lines:
            if ': ' in line:
                key, value = line.split(': ', 1)
                if key in patient_data:
                    patient_data[key] = value
        data.append(patient_data)

# Write to CSV
output_file = '/home/luciacev/Desktop/LLM/dashboard_output_final_paul/patient_data.csv'
with open(output_file, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=all_keys)
    writer.writeheader()
    writer.writerows(data)

print(f"CSV created at {output_file}")