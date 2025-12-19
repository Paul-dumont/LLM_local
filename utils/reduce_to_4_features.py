#!/usr/bin/env python3
"""
Réduire les features de 46 à 4 en gardant uniquement les features liées aux maux de tête
À partir de data_output_clean_46, crée data_output_clean_4_headache
"""
from pathlib import Path

# Features à garder (4 au total)
FEATURES_TO_KEEP = {
    "headache_intensity",
    "headache_location",
    "headache_frequency",
    "migraine_history",
}

def keep_only_features(text: str, features_to_keep: set) -> str:
    """
    Garde uniquement les lignes contenant les features spécifiées.
    Format attendu: "feature_name: value"
    """
    lines = text.strip().split('\n')
    filtered_lines = []
    
    for line in lines:
        # Extraire le nom de la feature (avant le ':')
        if ':' in line:
            feature_name = line.split(':')[0].strip()
            if feature_name in features_to_keep:
                filtered_lines.append(line)
        else:
            # Lignes sans ':' sont conservées si c'est patient_id
            if line.strip() and line.strip().startswith("patient_id"):
                filtered_lines.append(line)
    
    # S'assurer que patient_id est en premier
    result_lines = []
    patient_id_line = None
    
    for line in filtered_lines:
        if line.strip().startswith("patient_id"):
            patient_id_line = line
        else:
            result_lines.append(line)
    
    if patient_id_line:
        result_lines.insert(0, patient_id_line)
    
    return '\n'.join(result_lines)

def main():
    BASE_DIR = Path(__file__).parent.parent
    INPUT_DIR = BASE_DIR / "data_training" / "data_output_clean_46"
    OUTPUT_DIR = BASE_DIR / "data_training" / "data_output_clean_4_headache"
    
    # Créer le dossier de sortie
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not INPUT_DIR.exists():
        print(f"❌ Dossier d'entrée non trouvé: {INPUT_DIR}")
        return
    
    # Lister tous les fichiers
    input_files = sorted(INPUT_DIR.glob("*.txt"))
    print(f"📊 RÉDUCTION: 46 → 4 features (Headache)")
    print(f"📂 Traitement de {len(input_files)} fichiers...")
    print(f"📁 Source: {INPUT_DIR}")
    print(f"📁 Destination: {OUTPUT_DIR}")
    print(f"🎯 Features à garder ({len(FEATURES_TO_KEEP)}):")
    for i, feature in enumerate(sorted(FEATURES_TO_KEEP), 1):
        print(f"   {i}. {feature}")
    print()
    
    count = 0
    for input_file in input_files:
        # Lire le fichier
        content = input_file.read_text(encoding="utf-8")
        
        # Garder uniquement les features spécifiées
        filtered_content = keep_only_features(content, FEATURES_TO_KEEP)
        
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
    print(f"   📊 Features conservées: 4/46")
    print("="*70)

if __name__ == "__main__":
    main()
