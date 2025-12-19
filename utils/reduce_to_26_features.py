#!/usr/bin/env python3
"""
Réduire les features de 46 à 26 en supprimant 20 critères spécifiques
À partir de data_output_clean_46, crée data_output_clean_26
"""
from pathlib import Path

# Features à supprimer (20 au total - taux de représentation < 50%)
FEATURES_TO_REMOVE = {
    "headache_frequency",           # 48.80%
    "sleep_disorder_type",          # 47.20%
    "maximum_opening_without_pain", # 46.60%
    "neck_pain_present",            # 39.00%
    "current_appliance",            # 38.00%
    "onset_triggers",               # 27.00%
    "physical_therapy_status",      # 24.80%
    "adverse_reactions",            # 24.80%
    "jaw_crepitus",                 # 22.40%
    "jaw_locking",                  # 14.60%
    "pain_relieving_factors",       # 13.40%
    "back_pain_present",            # 12.60%
    "sleep_apnea_diagnosed",        # 12.40%
    "autoimmune_condition",         # 9.20%
    "migraine_history",             # 7.20%
    "previous_medications",         # 7.00%
    "pain_frequency",               # 4.60%
    "depression_present",           # 3.60%
    "pain_duration",                # 2.60%
    "fibromyalgia_present",         # 2.40%
}

def remove_features(text: str, features_to_remove: set) -> str:
    """
    Supprime les lignes contenant les features spécifiées.
    Format attendu: "feature_name: value"
    """
    lines = text.strip().split('\n')
    filtered_lines = []
    
    for line in lines:
        # Extraire le nom de la feature (avant le ':')
        if ':' in line:
            feature_name = line.split(':')[0].strip()
            if feature_name not in features_to_remove:
                filtered_lines.append(line)
        else:
            # Lignes sans ':' sont conservées (commentaires, etc)
            if line.strip():
                filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)

def main():
    BASE_DIR = Path(__file__).parent.parent
    INPUT_DIR = BASE_DIR / "data_training" / "data_output_clean_46"
    OUTPUT_DIR = BASE_DIR / "data_training" / "data_output_clean_26"
    
    # Créer le dossier de sortie
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not INPUT_DIR.exists():
        print(f"❌ Dossier d'entrée non trouvé: {INPUT_DIR}")
        return
    
    # Lister tous les fichiers
    input_files = sorted(INPUT_DIR.glob("*.txt"))
    print(f"📊 RÉDUCTION: 46 → 26 features")
    print(f"📂 Traitement de {len(input_files)} fichiers...")
    print(f"📁 Source: {INPUT_DIR}")
    print(f"📁 Destination: {OUTPUT_DIR}")
    print(f"🗑️  Features à supprimer ({len(FEATURES_TO_REMOVE)}):")
    for i, feature in enumerate(sorted(FEATURES_TO_REMOVE), 1):
        print(f"   {i:2d}. {feature}")
    print()
    
    count = 0
    for input_file in input_files:
        # Lire le fichier
        content = input_file.read_text(encoding="utf-8")
        
        # Supprimer les features
        filtered_content = remove_features(content, FEATURES_TO_REMOVE)
        
        # Sauvegarder dans le nouveau dossier
        output_file = OUTPUT_DIR / input_file.name
        output_file.write_text(filtered_content, encoding="utf-8")
        
        count += 1
        if count % 50 == 0 or count == len(input_files):
            print(f"✅ Traité {count}/{len(input_files)}")
    
    print()
    print("="*70)
    print(f"✅ TERMINÉ!")
    print(f"   {count} fichiers traités")
    print(f"   📁 Résultats: {OUTPUT_DIR}")
    print(f"   📊 Features conservées: 26/46")
    print("="*70)

if __name__ == "__main__":
    main()
