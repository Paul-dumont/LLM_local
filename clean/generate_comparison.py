#!/usr/bin/env python3
"""
Génère un fichier JSON avec toutes les comparaisons pour l'interface web
"""

import json
import os

# Répertoires (paths relative to repository root)
# Run this script from the repository root (/home/luciacev/Desktop/LLM)
DATA_OUTPUT_12_DIR = 'data_training/data_output_clean'
REFERENCE_DIR = 'data_training/data_output_clea'
PREDICTIONS_12_DIR = 'Qwen2.5-7B-instruct/predict_Qwen7B_all'
LLM_PREDICTIONS_DIR = 'Qwen2.5-7B-instruct/predict_Qwen7B_all'

def parse_file(filepath):
    """Parser un fichier summary en préservant l'ordre"""
    result = {}
    keys_order = []  # Pour garder l'ordre
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        result[key] = parts[1].strip()
                        if key not in keys_order:
                            keys_order.append(key)
    except Exception as e:
        print(f"Erreur lecture {filepath}: {e}")
    return result, keys_order

def generate_comparison_data():
    """Générer toutes les comparaisons avec 3 colonnes"""
    
    # Lister tous les fichiers de prédictions LLM
    llm_files = sorted([f for f in os.listdir(LLM_PREDICTIONS_DIR) 
                   if f.endswith('_pred.txt')])
    
    # Vérifier les dossiers de référence
    reference_files = set()
    if os.path.exists(REFERENCE_DIR):
        reference_files = set([f for f in os.listdir(REFERENCE_DIR) 
                             if f.endswith('_summary.txt')])
    
    data_output_12_files = set()
    if os.path.exists(DATA_OUTPUT_12_DIR):
        data_output_12_files = set([f for f in os.listdir(DATA_OUTPUT_12_DIR) 
                             if f.endswith('_summary.txt') or f.endswith('.txt')])
    
    predictions_12_files = set()
    if os.path.exists(PREDICTIONS_12_DIR):
        predictions_12_files = set([f for f in os.listdir(PREDICTIONS_12_DIR) 
                             if f.endswith('_pred.txt')])
    
    all_comparisons = {}
    files_compared = 0
    files_missing = 0
    
    print(f"Génération des comparaisons pour {len(llm_files)} fichiers de prédictions...")
    print(f"Fichiers référence (harmonisés) disponibles: {len(reference_files)}")
    print(f"Fichiers Data_output_12 disponibles: {len(data_output_12_files)}")
    print(f"Fichiers Predictions_12 disponibles: {len(predictions_12_files)}")
    
    for filename in llm_files:
        llm_path = os.path.join(LLM_PREDICTIONS_DIR, filename)
        
        # Chercher les fichiers correspondants
        # Format: B001_pred.txt -> B001_summary.txt
        patient_id = filename.replace('_pred.txt', '')
        reference_filename = f"{patient_id}_summary.txt"
        reference_path = os.path.join(REFERENCE_DIR, reference_filename)
        
        # Chercher Data_output_12
        data_output_12_path = None
        possible_names = [f"{patient_id}_summary.txt", f"{patient_id}.txt", f"{patient_id}_Word_text.txt"]
        for name in possible_names:
            test_path = os.path.join(DATA_OUTPUT_12_DIR, name)
            if os.path.exists(test_path):
                data_output_12_path = test_path
                break
        
        # Chercher Predictions_12
        predictions_12_path = None
        predictions_12_filename = f"{patient_id}_pred.txt"
        predictions_12_path = os.path.join(PREDICTIONS_12_DIR, predictions_12_filename)
        if not os.path.exists(predictions_12_path):
            predictions_12_path = None
        
        # Parser les quatre fichiers
        reference_data = {}
        reference_order = []
        llm_data, llm_order = parse_file(llm_path)
        data_output_12_data = {}
        data_output_12_order = []
        predictions_12_data = {}
        predictions_12_order = []
        
        if os.path.exists(reference_path):
            reference_data, reference_order = parse_file(reference_path)
            files_compared += 1
        else:
            files_missing += 1
        
        if data_output_12_path:
            data_output_12_data, data_output_12_order = parse_file(data_output_12_path)
        
        if predictions_12_path:
            predictions_12_data, predictions_12_order = parse_file(predictions_12_path)
        
        # Utiliser l'ordre du fichier référence
        keys_order = reference_order if reference_order else llm_order
        
        # Ajouter les clés manquantes à la fin
        all_keys_set = set(llm_data.keys()) | set(reference_data.keys()) | set(data_output_12_data.keys()) | set(predictions_12_data.keys())
        for key in all_keys_set:
            if key not in keys_order:
                keys_order.append(key)
        
        comparison = []
        
        for key in keys_order:
            data_output_12_val = data_output_12_data.get(key, '')
            reference_val = reference_data.get(key, '')
            predictions_12_val = predictions_12_data.get(key, '')
            llm_val = llm_data.get(key, '')
            
            comparison.append({
                'key': key,
                'data_output_12': data_output_12_val,
                'predictions_12': predictions_12_val,
                'reference': reference_val,
                'llm_prediction': llm_val
            })
        
        all_comparisons[f"{patient_id}_pred.txt"] = {
            'comparison': comparison,
            'stats': {
                'total': len(comparison),
                'reference_available': os.path.exists(reference_path),
                'data_output_12_available': data_output_12_path is not None,
                'predictions_12_available': predictions_12_path is not None
            }
        }
    
    # Sauvegarder le JSON
    output_file = 'clean/comparison_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_comparisons, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Fichier généré: {output_file}")
    print(f"✓ {files_compared} fichiers avec références comparés")
    print(f"⚠ {files_missing} fichiers sans références harmonisées")
    print(f"✓ Total: {len(all_comparisons)} fichiers dans le JSON")
    print(f"\nVous pouvez maintenant ouvrir 'compare_patients.html' avec Live Server")

if __name__ == '__main__':
    generate_comparison_data()
