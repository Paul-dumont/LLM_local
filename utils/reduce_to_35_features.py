#!/usr/bin/env python3
"""
Réduire les features de 46 à 35 en supprimant 11 critères spécifiques
À partir de data_output_clean_46, crée data_output_clean_35
"""
import re
from pathlib import Path

# Features à supprimer (11 au total)
FEATURES_TO_REMOVE = {
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
    "fibromyalgia_present",
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
    OUTPUT_DIR = BASE_DIR / "data_training" / "data_output_clean_35"
    
    # Créer le dossier de sortie
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not INPUT_DIR.exists():
        print(f"❌ Dossier d'entrée non trouvé: {INPUT_DIR}")
        return
    
    # Lister tous les fichiers
    input_files = sorted(INPUT_DIR.glob("*.txt"))
    print(f"📂 Traitement de {len(input_files)} fichiers...")
    print(f"📁 Source: {INPUT_DIR}")
    print(f"📁 Destination: {OUTPUT_DIR}")
    print(f"🗑️  Features à supprimer ({len(FEATURES_TO_REMOVE)}):")
    for feature in sorted(FEATURES_TO_REMOVE):
        print(f"   - {feature}")
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
    print("="*70)

if __name__ == "__main__":
    main()
