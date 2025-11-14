#!/usr/bin/env python3
"""
Script d'harmonisation des summaries sans dépendances externes (pas de LLM)
Version simple et rapide basée uniquement sur des règles
"""

import json
import os
import re
from typing import Dict, List
from collections import defaultdict

# Charger les valeurs acceptées
with open('../value.json', 'r', encoding='utf-8') as f:
    value_data = json.load(f)
    indicators = {feature['name']: feature['possible_output'] for feature in value_data['features']}

# Répertoires
# input_dir = '../Qwen1.5B_full_V3_local/predictions_12'
# output_dir = '../Qwen1.5B_full_V3_local/predictions_12_harmonized'
input_dir = '../Data_ouput_Bart'
output_dir = '../Data_ouput_Bart_harmonized'
os.makedirs(output_dir, exist_ok=True)

# Dictionnaire de synonymes pour améliorer la correspondance
SYNONYMES = {
    # === BOOLÉENS ===
    'yes': 'true', 'oui': 'true', 'present': 'true', 'y': 'true', 'positive': 'true',
    'no': 'false', 'non': 'false', 'absent': 'false', 'n': 'false', 'negative': 'false',
    'alse': 'false', 'fase': 'false', 'falase': 'false',  # typos
    'ture': 'true', 'rue': 'true',  # typos
    
    # === TMJ LOCALISATIONS ===
    'left': 'left TMJ', 'right': 'right TMJ', 
    'bilateral': 'left and right TMJ', 'both': 'left and right TMJ', 
    'both sides': 'left and right TMJ',
    'left tmj': 'left TMJ', 'right tmj': 'right TMJ',
    'left mj': 'left TMJ',  # typo
    'tmj': 'left and right TMJ',  # si juste "tmj" sans précision
    
    # === TEMPS/PÉRIODES ===
    'am': 'morning', 'morning': 'morning', 'matin': 'morning',
    'mornings': 'morning', 'in the morning': 'morning', 'in the mornings': 'morning',
    'pm': 'afternoon', 'afternoon': 'afternoon', 'après-midi': 'afternoon',
    'afternoons': 'afternoon', 'in the afternoon': 'afternoon',
    'evening': 'evening', 'soir': 'evening', 'night': 'evening',
    'evenings': 'evening', 'in the evening': 'evening', 'at night': 'evening',
    
    # === FRÉQUENCES ===
    'daily': 'daily', 'every day': 'daily', 'quotidien': 'daily',
    'constant': 'constant', 'continuous': 'continuous', 'continual': 'continuous',
    'occasional': 'occasional', 'sometimes': 'occasional', 'periodic': 'occasional',
    'once a week': 'once a week', 'weekly': 'once a week',
    'once a month': 'once a month', 'monthly': 'once a month',
    '1 time per week': '1 a week', '1 per week': '1 a week',
    '2 per week': '2 a week', '2 times a week': '2 a week',
    '2 times a month': '2 a month',
    
    # === SÉVÉRITÉ/INTENSITÉ ===
    'mild': 'mild', 'léger': 'mild', 'light': 'mild',
    'moderate': 'moderate', 'modéré': 'moderate', 'medium': 'moderate',
    'severe': 'severe', 'sévère': 'severe', 
    'high': 'high', 'high moderate': 'moderate',
    'extreme': 'severe', 'excruciating': 'severe', 'extremely high': 'severe',
    
    # === TMJ PAIN RATING (en dernier recours si pas de chiffre) ===
    'severe': '9', 'tmj pain severe': '9',
    'mild': '5', 'tmj pain mild': '5',
    
    # === LOCALISATIONS TÊTE/MAL DE TÊTE ===
    'frontal': 'frontal', 'front': 'frontal', 'forehead': 'frontal',
    'temporal': 'temporal', 'temple': 'temporal', 'temples': 'temporal',
    'anterior temple': 'temporal', 'anterior temporal': 'temporal',
    'posterior head': 'posterior head', 'back of head': 'posterior head',
    'back of the head': 'posterior head', 'occipital': 'posterior head',
    'behind eyes': 'behind eyes', 'behind her eyes': 'behind eyes', 'behind the eyes': 'behind eyes',
    'around the eyes': 'behind eyes',
    'top of the head': 'top of the head', 'top of her head': 'top of the head', 'top of head': 'top of the head',
    'throughout head': 'throughout head', 'entire head': 'throughout head', 'whole head': 'throughout head',
    'all over head': 'throughout head', 'all over her head': 'throughout head',
    'behind ears': 'posterior head',
    
    # === MUSCLES ===
    'masseter': 'masseter', 'masseters': 'masseter',
    'bilateral masseters': 'masseter',
    'anterior band of the masseter': 'masseter',
    'temporalis': 'temporalis', 'bilateral temporalis': 'temporalis',
    'neck': 'neck', 'cou': 'neck', 'posterior neck': 'neck',
    'shoulders': 'shoulder', 'shoulder': 'shoulder', 'bilateral shoulders': 'shoulder',
    'styloid': 'styloid', 'bilateral styloid areas': 'styloid',
    'retromandibular': 'retromandibular',
    'scm': 'scm', 'scms': 'scm', 'sternocleidomastoid': 'scm',
    'head': 'temporalis',  # approximation
    'head and neck area': 'neck',
    
    # === ARTICULATIONS/JOINTS ===
    'knees': 'knees', 'knee': 'knees', 'genoux': 'knees',
    'left knee': 'knees', 'right knee': 'knees',
    'hips': 'hips', 'hip': 'hips', 'hanches': 'hips',
    'left hip': 'hips', 'right hip': 'hips',
    'back': 'back', 'dos': 'back', 'spine': 'back',
    'wrists': 'wrists', 'wrist': 'wrists', 'poignets': 'wrists', 'poignet': 'wrists',
    'left wrist': 'wrists', 'right wrist': 'wrists',
    'ankles': 'ankles', 'ankle': 'ankles', 'chevilles': 'ankles', 'cheville': 'ankles',
    'left ankle': 'ankles', 'right ankle': 'ankles',
    'hands': 'hands', 'hand': 'hands', 'mains': 'hands', 'main': 'hands',
    'left hand': 'hands', 'right hand': 'hands',
    'elbows': 'elbows', 'elbow': 'elbows', 'coudes': 'elbows', 'coude': 'elbows',
    'left elbow': 'elbows', 'right elbow': 'elbows',

    
    # === APPAREILS DENTAIRES ===
    'splint': 'splint', 'gouttière': 'splint', 'orthotic': 'splint',
    'bite splint': 'splint', 'bite splints': 'splint',
    'occlusal splint': 'splint', 'occlusal splints': 'splint',
    'night guard': 'night guard', 'night guards': 'night guard',
    'orthodontics': 'orthodontics', 'orthodontie': 'orthodontics', 'orthodontic': 'orthodontics',
    'orthodontic': 'orthodontics', 'orthodontic treatment': 'orthodontics',
    'clear aligner': 'orthodontics', 'clear aligner therapy': 'orthodontics',
    'retainer': 'retainer', 'contenteur': 'retainer',
    'essex retainer': 'retainer', 'essex retainers': 'retainer',
    
    # === CONDITIONS/SYMPTÔMES ===
    'bruxism': 'bruxism', 'grinding': 'bruxism', 'grincement': 'bruxism',
    'brux': 'bruxism', 'bruxing': 'bruxism', 'brxism': 'bruxism',  # typos/variations
    'clenching': 'clenching', 'serrement': 'clenching',
    'clench': 'clenching',
    'snoring': 'snoring', 'ronflement': 'snoring',
    'sleep apnea': 'sleep apnea', 'apnea': 'sleep apnea', 'apnée': 'sleep apnea',
    'mild sleep apnea': 'sleep apnea', 'severe sleep apnea': 'sleep apnea',
    'somnolence': 'somnolence', 'daytime somnolence': 'somnolence',
    'daytime tiredness': 'somnolence', 'dayteim somnolence': 'somnolence',  # typo
    'chronic fatigue': 'somnolence',
    'mouth breathing': 'snoring',  # approximation
    'gasps for air': 'sleep apnea',
    
    # === DÉCLENCHEURS/TRIGGERS ===
    'mva': 'motor vehicle accident', 
    'car accident': 'motor vehicle accident',
    'motor vehicle accident': 'motor vehicle accident', 
    'vehicle accident': 'motor vehicle accident', 
    'auto accident': 'motor vehicle accident',
    'car': 'motor vehicle accident', 'accident': 'motor vehicle accident',
    'motorcycle': 'motor vehicle accident', 'motorcycle accident': 'motor vehicle accident',
    'vehicle': 'motor vehicle accident', 'motorbike': 'motor vehicle accident',
    'motorbike accident': 'motor vehicle accident',
    'orthodontic': 'orthodontics', 'orthodontics': 'orthodontics',
    'orthodontic treatment': 'orthodontics', 'elastics': 'orthodontics',
    'retainer': 'orthodontics', 'orthodontic retainer': 'orthodontics',
    'braces': 'orthodontics',
    'other': 'other',
    
    # === ALLERGIES/RÉACTIONS ADVERSES ===
    'penicillin': 'penicillin', 'pénicilline': 'penicillin',
    'codeine': 'codeine', 'codéine': 'codeine',
    'seasonal': 'seasonal', 'seasonal allergies': 'seasonal',
    'pollen': 'pollen', 'pollens': 'pollen', 'pollen allergies': 'pollen',
    'dust': 'dust', 'dust allergies': 'dust',
    'environmental': 'environmental', 'environmental allergies': 'environmental',
    'animal': 'animal', 'animals': 'animal', 'animal allergies': 'animal',
    'dog': 'animal', 'dogs': 'animal',
    'cat': 'animal', 'cats': 'animal',
    'horse': 'animal', 'horses': 'animal',
    'none': 'none', 'no allergies': 'none',
    
    # === THÉRAPIES ===
    'physical therapy': 'physical therapy', 'pt': 'physical therapy',
    'physiotherapy': 'physical therapy', 'physio': 'physical therapy',
    'physical therapy treatment': 'physical therapy',
    'chiropractor': 'chiropratic', 'chiropractic': 'chiropratic', 
    'chiro': 'chiropratic', 'chiropractic treatment': 'chiropratic',
    'chiropratic': 'chiropratic', 'chiropratic treatment': 'chiropratic',
    'acupuncture': 'physical therapy',  # on classe dans physical therapy
    'acupressure': 'physical therapy',
    'massage': 'physical therapy', 'massage therapy': 'physical therapy',
    
    # === DISC DISPLACEMENT ===
    'with reduction': 'with reduction', 
    'without reduction': 'without reduction',
    'disc dislocation': 'left disc dislocation',  # si pas de côté précisé
    'disc displacement': 'left disc dislocation',
    'left disc': 'left disc dislocation', 
    'right disc': 'right disc dislocation',
    'left disc dislocation': 'left disc dislocation',
    'right disc dislocation': 'right disc dislocation',
    'left articular disc': 'left disc dislocation',
    'right articular disc': 'right disc dislocation',
    'left anterior displacement': 'left disc dislocation',
    'right anterior displacement': 'right disc dislocation',
    'left anteriorly displaced': 'left disc dislocation',
    'right anteriorly displaced': 'right disc dislocation',
    
    # === CONDITIONS AUTO-IMMUNES ===
    'rheumatoid arthritis': 'rheumatoid arthritis', 'ra': 'rheumatoid arthritis',
    'psoriasis': 'psoriasis',
    'hashimoto': 'hashimoto thyoiditis', 
    "hashimoto's disease": 'hashimoto thyoiditis',
    'hashimoto thyroiditis': 'hashimoto thyoiditis',
    
    # === CROSSBITE ===
    'anterior open bite': 'anterior open bite',
    'anterior open bite tendency': 'anterior open bite',
    'posterior cross bite': 'posterior cross bite',
    'posterior crossbite': 'posterior cross bite',
    'bilateral posterior crossbite': 'posterior cross bite',
    'buccal crossbite': 'posterior cross bite',
    'left posterior crossbite': 'posterior cross bite',
    'right posterior crossbite': 'posterior cross bite',
    'left posterior crossbite tendency': 'posterior cross bite',
    'right posterior crossbite tendency': 'posterior cross bite',
    'posterior crossbite tendency': 'posterior cross bite',
}



def nettoyer_texte(texte: str) -> str:
    """Nettoyer et normaliser le texte"""
    return texte.strip().lower()


def extraire_nombre(texte: str) -> List[int]:
    """Extraire tous les nombres d'un texte"""
    return [int(n) for n in re.findall(r'\d+', texte)]


# Indicateurs qui peuvent avoir plusieurs valeurs
MULTI_VALUE_INDICATORS = [
    'pain_aggravating_factors',
    'muscle_pain_location',
    'joint_pain_areas',
    'sleep_disorder_type',
    'disc_displacement',
    'headache_location',
    'pain_relieving_factors',
    'adverse_reactions',
    'previous_medications',
    'current_medications',
    'autoimmune_condition',
    'onset_triggers',
    'physical_therapy_status',
    'appliance_history',
    'current_appliance',
    'jaw_clicking',
    'jaw_crepitus',
    'joint_arthritis_location'
]


def normaliser_valeur(valeur_brute: str, options_autorisees: List[str], nom_indicateur: str = "") -> str:
    """
    Normaliser une valeur brute pour qu'elle corresponde à une des options autorisées
    Retourne une seule valeur OU plusieurs valeurs séparées par " | " si l'indicateur le permet
    
    RÈGLE SPÉCIALE: Si "whatever" est dans les options possibles :
    1. Essayer de matcher avec les valeurs spécifiques
    2. Si "unknown" dans l'input → retourner "unknown"
    3. Sinon → retourner la valeur d'origine telle quelle
    """
    if not valeur_brute or valeur_brute.strip() == "":
        return "unknown"
    
    valeur_originale = valeur_brute
    valeur_nettoyee = nettoyer_texte(valeur_brute)
    # Normalisations rapides : préférer la forme chiffrée pour "once a week/month"
    # (doit être faites tôt pour prendre effet avant les vérifications exactes)
    valeur_nettoyee = valeur_nettoyee.replace('once a week', '1 a week').replace('once a month', '1 a month')
    has_whatever = 'whatever' in options_autorisees
    
    # Vérifier si cet indicateur accepte plusieurs valeurs
    allow_multiple = nom_indicateur in MULTI_VALUE_INDICATORS
    
    # === CAS SPÉCIAL PRIORITAIRE : current_medications - nettoyer "none" des listes ===
    if nom_indicateur == 'current_medications' and ('|' in valeur_brute or ',' in valeur_brute):
        # Séparer les médicaments et filtrer "none"
        separateurs = re.compile(r'\s*[|,]\s*', re.IGNORECASE)
        parties = separateurs.split(valeur_brute)
        parties_filtrees = [p.strip() for p in parties if p.strip() and nettoyer_texte(p.strip()) != 'none']
        
        if parties_filtrees:
            valeur_originale = ' | '.join(parties_filtrees)
            valeur_brute = valeur_originale
            valeur_nettoyee = nettoyer_texte(valeur_brute)
    
    # === CAS SPÉCIAUX PAR INDICATEUR ===
    
    # Patient ID
    if nom_indicateur == "patient_id":
        match = re.search(r'B\d+', valeur_brute, re.IGNORECASE)
        if match:
            return match.group(0).upper()
        return "unknown"
    
    # Mesures en mm (maximum_opening, maximum_opening_without_pain)
    if nom_indicateur in ['maximum_opening_without_pain', 'maximum_opening']:
        # Si c'est explicitement "unknown", le garder
        if valeur_nettoyee == 'unknown':
            return 'unknown'
        
        match = re.search(r'(\d+)\s*mm', valeur_brute, re.IGNORECASE)
        if match:
            return f"{match.group(1)}mm"
        # Si pas de nombre trouvé mais XXmm dans les options
        if 'XXmm' in options_autorisees:
            return "XXmm"
        return "unknown"
    
    # Âge (0-100)
    if nom_indicateur == 'patient_age':
        nombres = extraire_nombre(valeur_brute)
        if nombres:
            age = nombres[0]
            if 0 <= age <= 100:
                return str(age)
        return "unknown"
    
    # Scores (0-10)
    if nom_indicateur in ['diet_score', 'disability_rating', 'jaw_function_score',
                          'average_daily_pain_intensity', 'headache_intensity', 'tmj_pain_rating']:
        nombres = extraire_nombre(valeur_brute)
        if nombres:
            # Filtrer les nombres valides (0-10)
            nombres_valides = [n for n in nombres if 0 <= n <= 10]
            if nombres_valides:
                # Calculer la moyenne et arrondir
                moyenne = sum(nombres_valides) / len(nombres_valides)
                score = round(moyenne)
                return str(score)
        
        # Si pas de nombre trouvé, vérifier les synonymes (seulement pour tmj_pain_rating)
        if nom_indicateur == 'tmj_pain_rating':
            if valeur_nettoyee in SYNONYMES:
                return SYNONYMES[valeur_nettoyee]
        
        return "unknown"
    
    # Date d'apparition de la douleur (X years ago / X years of age)
    if nom_indicateur == 'pain_onset_date':
        total_years = 0.0
        
        # Convertir "a year" et "one year" en chiffres
        valeur_temp = valeur_nettoyee
        valeur_temp = valeur_temp.replace('a year', '1 year').replace('one year', '1 year')
        valeur_temp = valeur_temp.replace('two years', '2 years').replace('three years', '3 years')
        valeur_temp = valeur_temp.replace('four years', '4 years').replace('five years', '5 years')
        valeur_temp = valeur_temp.replace('six years', '6 years').replace('seven years', '7 years')
        valeur_temp = valeur_temp.replace('eight years', '8 years').replace('nine years', '9 years')
        valeur_temp = valeur_temp.replace('ten years', '10 years')
        
        # Convertir les mots en chiffres pour les mois aussi
        valeur_temp = valeur_temp.replace('one month', '1 month').replace('two months', '2 months')
        valeur_temp = valeur_temp.replace('three months', '3 months').replace('four months', '4 months')
        valeur_temp = valeur_temp.replace('five months', '5 months').replace('six months', '6 months')
        valeur_temp = valeur_temp.replace('seven months', '7 months').replace('eight months', '8 months')
        valeur_temp = valeur_temp.replace('nine months', '9 months').replace('ten months', '10 months')
        valeur_temp = valeur_temp.replace('eleven months', '11 months').replace('twelve months', '12 months')
        
        # Chercher les années
        match_years = re.search(r'(\d+)\s*years?', valeur_temp, re.IGNORECASE)
        if match_years:
            total_years += int(match_years.group(1))
        
        # Chercher les mois et les convertir en fraction d'années (avec décimales)
        match_months = re.search(r'(\d+)\s*months?', valeur_temp, re.IGNORECASE)
        if match_months:
            mois = int(match_months.group(1))
            total_years += round(mois / 12, 1)  # Arrondir à 1 décimale
        
        # Si on a trouvé au moins des années ou des mois
        if match_years or match_months:
            # Formater le nombre (enlever .0 si c'est un entier)
            if total_years == int(total_years):
                total_years_str = str(int(total_years))
            else:
                total_years_str = str(total_years)
            
            # Détecter "ago" → X years ago
            if 'ago' in valeur_nettoyee:
                return f"{total_years_str} years ago"
            # Détecter "age" ou "old" → X years of age
            elif 'age' in valeur_nettoyee or 'old' in valeur_nettoyee:
                return f"{total_years_str} years of age"
        
        # Sinon chercher un nombre simple
        match_nombre = re.search(r'(\d+)', valeur_brute)
        if match_nombre:
            nombre = match_nombre.group(1)
            # Détecter "ago" → X years ago
            if 'ago' in valeur_nettoyee:
                return f"{nombre} years ago"
            # Détecter "age" ou "old" → X years of age
            elif 'age' in valeur_nettoyee or 'old' in valeur_nettoyee:
                return f"{nombre} years of age"
            else:
                # Si on a un nombre seul, par défaut "years of age"
                return f"{nombre} years of age"
        
        return "unknown"
    
    # Fréquence des maux de tête (X a week, W a month, etc.)
    if nom_indicateur == 'headache_frequency':
        # D'abord chercher les correspondances exactes
        for option in options_autorisees:
            if option.lower() in valeur_nettoyee:
                return option
        
        # Convertir nombres en lettres vers chiffres pour la détection
        valeur_temp = valeur_nettoyee
        # explicit conversions for common phrases
        valeur_temp = valeur_temp.replace('once a week', '1 a week').replace('once a month', '1 a month')
        valeur_temp = valeur_temp.replace('once', '1').replace('twice', '2').replace('one', '1').replace('two', '2').replace('three', '3')
        valeur_temp = valeur_temp.replace('four', '4').replace('five', '5').replace('six', '6')
        valeur_temp = valeur_temp.replace('seven', '7').replace('eight', '8').replace('nine', '9')
        
        # Chercher pattern avec nombre + week(s)
        match_week = re.search(r'(\d+).*?weeks?', valeur_temp, re.IGNORECASE)
        if match_week:
            return f"{match_week.group(1)} a week"
        
        # Chercher pattern avec nombre + month(s)
        match_month = re.search(r'(\d+).*?months?', valeur_temp, re.IGNORECASE)
        if match_month:
            return f"{match_month.group(1)} a month"
        
        # Chercher "once" ou "1 time" pour détecter "1 time per week"
        if '1' in valeur_temp and ('week' in valeur_temp or 'weekly' in valeur_temp):
            return "1 time per week"
        if '1' in valeur_temp and ('month' in valeur_temp or 'monthly' in valeur_temp):
            return "1 time per month"
        
        # Dernier recours : si "week" ou "weeks" détecté sans nombre → weekly
        if 'week' in valeur_nettoyee or 'weeks' in valeur_nettoyee:
            if 'weekly' in options_autorisees:
                return 'weekly'
        
        # Dernier recours avant unknown : si "day" ou "days" → daily
        if 'day' in valeur_nettoyee or 'days' in valeur_nettoyee:
            if 'daily' in options_autorisees:
                return 'daily'
        
        return "unknown"
    
    # Fréquence de la douleur (pain_frequency)
    if nom_indicateur == 'pain_frequency':
        # D'abord chercher les correspondances exactes
        for option in options_autorisees:
            if option.lower() in valeur_nettoyee:
                return option
        
        # Convertir nombres en lettres vers chiffres pour la détection
        valeur_temp = valeur_nettoyee
        # explicit conversions for common phrases
        valeur_temp = valeur_temp.replace('once a week', '1 a week').replace('once a month', '1 a month')
        valeur_temp = valeur_temp.replace('once', '1').replace('twice', '2').replace('one', '1').replace('two', '2').replace('three', '3')
        valeur_temp = valeur_temp.replace('four', '4').replace('five', '5').replace('six', '6')
        valeur_temp = valeur_temp.replace('seven', '7').replace('eight', '8').replace('nine', '9')
        
        # Chercher pattern avec nombre + week(s)
        match_week = re.search(r'(\d+).*?weeks?', valeur_temp, re.IGNORECASE)
        if match_week:
            return f"{match_week.group(1)} a week"
        
        # Chercher pattern avec nombre + month(s)
        match_month = re.search(r'(\d+).*?months?', valeur_temp, re.IGNORECASE)
        if match_month:
            return f"{match_month.group(1)} a month"
        
        # Chercher "once" ou "1 time" pour détecter "1 time per week"
        if '1' in valeur_temp and ('week' in valeur_temp or 'weekly' in valeur_temp):
            return "1 time per week"
        if '1' in valeur_temp and ('month' in valeur_temp or 'monthly' in valeur_temp):
            return "1 time per month"
        
        # Dernier recours : si "week" ou "weeks" détecté sans nombre → weekly
        if 'week' in valeur_nettoyee or 'weeks' in valeur_nettoyee:
            if 'weekly' in options_autorisees:
                return 'weekly'
        
        # Dernier recours avant unknown : si "day" ou "days" → daily
        if 'day' in valeur_nettoyee or 'days' in valeur_nettoyee:
            if 'daily' in options_autorisees:
                return 'daily'
        
        return "unknown"
    
    # Déclencheurs (onset_triggers) - gestion spéciale pour "mva"
    if nom_indicateur == 'onset_triggers':
        # Détection de mots-clés pour motor vehicle accident
        keywords_mva = ['mva', 'motor vehicle', 'car accident', 'car', 'accident', 'motorcycle', 'vehicle', 'motorbike', 'auto']
        if any(keyword in valeur_nettoyee for keyword in keywords_mva):
            return 'motor vehicle accident'
        # Détection de mots-clés pour orthodontics
        keywords_ortho = ['orthodontic', 'orthodontie', 'elastics', 'retainer', 'braces']
        if any(keyword in valeur_nettoyee for keyword in keywords_ortho):
            return 'orthodontics'
        # Sinon chercher dans les options
        for option in options_autorisees:
            if option.lower() in valeur_nettoyee:
                return option
        return 'other' if 'other' in options_autorisees else 'unknown'
    
    # Sévérité de la douleur musculaire
    if nom_indicateur == 'muscle_pain_score':
        # Détecter tous les niveaux présents et les retourner
        niveaux_trouves = []
        if 'minimal' in valeur_nettoyee:
            niveaux_trouves.append('minimal')
        if 'mild' in valeur_nettoyee:
            niveaux_trouves.append('mild')
        if 'moderate' in valeur_nettoyee:
            niveaux_trouves.append('moderate')
        if 'high' in valeur_nettoyee:
            niveaux_trouves.append('high')
        if 'severe' in valeur_nettoyee:
            niveaux_trouves.append('severe')
        
        if niveaux_trouves:
            if len(niveaux_trouves) > 1:
                return ' | '.join(niveaux_trouves)
            else:
                return niveaux_trouves[0]
        return "unknown"
    
    # Localisation des maux de tête - détection spéciale pour "throughout head"
    if nom_indicateur == 'headache_location':
        # Si "head" est présent, vérifier s'il n'y a pas d'autres mots-clés spécifiques
        if 'head' in valeur_nettoyee:
            # Liste des mots-clés spécifiques qui excluraient "throughout head"
            specific_keywords = ['frontal', 'front', 'forehead', 'temporal', 'temple', 
                                'posterior', 'back', 'occipital', 'behind', 'top', 'ear']
            
            # Si aucun mot-clé spécifique n'est présent, c'est "throughout head"
            if not any(keyword in valeur_nettoyee for keyword in specific_keywords):
                return 'throughout head'
    
    # === VÉRIFICATION DES CORRESPONDANCES EXACTES ===
    for option in options_autorisees:
        if option in ['whatever', 'unknown']:  # Skip ces options pour l'instant
            continue
        if valeur_nettoyee == option.lower():
            return option
    
    # === VALEURS BOOLÉENNES ===
    if set(options_autorisees) == {'true', 'false', 'unknown'}:
        mots_positifs = ['true', 'yes', 'present', 'y', '1', 'positive']
        mots_negatifs = ['false', 'no', 'absent', 'n', '0', 'negative', 'none']
        
        mots = valeur_nettoyee.split()
        if any(mot in mots_positifs for mot in mots):
            return 'true'
        if any(mot in mots_negatifs for mot in mots):
            return 'false'
    
    # === CAS SPÉCIAL : current_medications - ne pas transformer automatiquement ===
    # Pour current_medications, on garde "none" uniquement si c'est la valeur exacte
    if nom_indicateur == 'current_medications':
        if valeur_nettoyee == 'none':
            return 'none'
        # Sinon on continue le traitement normal sans conversion automatique
        # On ne transforme RIEN automatiquement pour current_medications
    
    # === VALEURS MULTIPLES (séparées par |, virgule, ou "and") ===
    if '|' in valeur_brute or ',' in valeur_brute or ' and ' in valeur_nettoyee:
        separateurs = re.compile(r'\s*[|,]\s*|\s+and\s+', re.IGNORECASE)
        parties = separateurs.split(valeur_brute)
        
        valeurs_trouvees = []
        
        for partie in parties:
            partie = partie.strip()
            if not partie:
                continue
            
            partie_nettoyee = nettoyer_texte(partie)
            
            # CAS SPÉCIAL pour current_medications : ignorer "none" dans les listes
            if nom_indicateur == 'current_medications' and partie_nettoyee == 'none':
                continue  # Sauter "none" s'il est dans une liste avec d'autres médicaments
            
            # CAS SPÉCIAL pour disc_displacement : détecter left/right + with/without reduction
            if nom_indicateur == 'disc_displacement':
                cote = None
                
                # Détecter le côté
                if 'left' in partie_nettoyee:
                    cote = 'left disc dislocation'
                elif 'right' in partie_nettoyee:
                    cote = 'right disc dislocation'
                
                # Ajouter le côté trouvé
                if cote and cote not in valeurs_trouvees:
                    valeurs_trouvees.append(cote)
                
                continue  # Passer à la partie suivante
            
            # Après avoir traité toutes les parties pour disc_displacement,
            # on ajoutera "with/without reduction" en fonction de la valeur complète
            
            # Correspondance exacte
            for option in options_autorisees:
                if option in ['whatever', 'unknown']:  # Skip
                    continue
                if partie_nettoyee == option.lower():
                    if option not in valeurs_trouvees:
                        valeurs_trouvees.append(option)
                    break
            else:
                # Essayer les synonymes
                if partie_nettoyee in SYNONYMES:
                    synonyme = SYNONYMES[partie_nettoyee]
                    if synonyme in options_autorisees and synonyme not in valeurs_trouvees:
                        valeurs_trouvees.append(synonyme)
                else:
                    # Vérifier si un mot de la partie correspond à un synonyme
                    trouve = False
                    for mot in partie_nettoyee.split():
                        if mot in SYNONYMES:
                            synonyme = SYNONYMES[mot]
                            if synonyme in options_autorisees and synonyme not in valeurs_trouvees:
                                valeurs_trouvees.append(synonyme)
                                trouve = True
                                break
                    
                    # Correspondance partielle si pas trouvé avec synonymes
                    if not trouve:
                        for option in options_autorisees:
                            if option in ['whatever', 'unknown']:  # Skip
                                continue
                            if option.lower() in partie_nettoyee or partie_nettoyee in option.lower():
                                if option not in valeurs_trouvees:
                                    valeurs_trouvees.append(option)
                                    break
        
        # Gestion spéciale pour disc_displacement : ajouter "with/without reduction" selon la valeur complète
        if nom_indicateur == 'disc_displacement' and valeurs_trouvees:
            # Vérifier si "reduction" apparaît dans la valeur complète
            if 'reduction' in valeur_nettoyee:
                if 'without reduction' in valeur_nettoyee:
                    if 'without reduction' not in valeurs_trouvees:
                        valeurs_trouvees.append('without reduction')
                else:
                    # Si juste "reduction" (ou "with reduction" ou "probable reduction")
                    if 'with reduction' not in valeurs_trouvees:
                        valeurs_trouvees.append('with reduction')
        
        # Gestion spéciale pour left/right TMJ -> left and right TMJ
        if 'left TMJ' in valeurs_trouvees and 'right TMJ' in valeurs_trouvees:
            if 'left and right TMJ' in options_autorisees:
                # Remplacer les deux par la version combinée
                valeurs_trouvees = [v for v in valeurs_trouvees if v not in ['left TMJ', 'right TMJ']]
                valeurs_trouvees.insert(0, 'left and right TMJ')
        
        # Si "left and right TMJ" existe déjà, supprimer "left TMJ" et "right TMJ" individuels
        if 'left and right TMJ' in valeurs_trouvees:
            valeurs_trouvees = [v for v in valeurs_trouvees if v not in ['left TMJ', 'right TMJ']]
        
        # Éliminer les doublons tout en préservant l'ordre
        valeurs_uniques = []
        for v in valeurs_trouvees:
            if v not in valeurs_uniques:
                valeurs_uniques.append(v)
        valeurs_trouvees = valeurs_uniques
        
        # CAS SPÉCIAL pour current_medications : filtrer "none" s'il y a d'autres valeurs
        if nom_indicateur == 'current_medications' and len(valeurs_trouvees) > 1:
            valeurs_trouvees = [v for v in valeurs_trouvees if v != 'none']
        
        # Retourner selon si l'indicateur accepte plusieurs valeurs
        if valeurs_trouvees:
            if allow_multiple and len(valeurs_trouvees) > 1:
                return ' | '.join(valeurs_trouvees)
            else:
                return valeurs_trouvees[0]
    
    # === APPLICATION DES SYNONYMES ===
    if valeur_nettoyee in SYNONYMES:
        synonyme = SYNONYMES[valeur_nettoyee]
        if synonyme in options_autorisees and synonyme not in ['whatever', 'unknown']:
            return synonyme
    
    # === CORRESPONDANCE FLOUE (fuzzy matching) ===
    # Pour les indicateurs multi-valeurs, on essaie de trouver toutes les correspondances
    if allow_multiple:
        valeurs_trouvees = []
        for option in options_autorisees:
            option_nettoyee = option.lower()
            
            # Ignorer les options génériques
            if option in ['whatever', 'unknown', 'none']:
                continue
            
            # Détection spéciale pour chiropratic (détecter "chiropractic" ou "chiropractor")
            if option == 'chiropratic':
                if 'chiropractic' in valeur_nettoyee or 'chiropractor' in valeur_nettoyee or 'chiro' in valeur_nettoyee:
                    if option not in valeurs_trouvees:
                        valeurs_trouvees.append(option)
                    continue
            
            # Détection spéciale pour motor vehicle accident (détecter mots-clés)
            if option == 'motor vehicle accident':
                keywords = ['car', 'accident', 'motorcycle', 'vehicle', 'motorbike', 'mva', 'auto']
                if any(keyword in valeur_nettoyee for keyword in keywords):
                    if option not in valeurs_trouvees:
                        valeurs_trouvees.append(option)
                    continue
            
            # Détection spéciale pour orthodontics (détecter mots-clés)
            if option == 'orthodontics':
                keywords = ['orthodontic', 'orthodontics', 'elastics', 'braces', 'clear aligner']
                if any(keyword in valeur_nettoyee for keyword in keywords):
                    if option not in valeurs_trouvees:
                        valeurs_trouvees.append(option)
                    continue
            
            # L'option est contenue dans la valeur
            if option_nettoyee in valeur_nettoyee:
                if option not in valeurs_trouvees:
                    valeurs_trouvees.append(option)
        
        # Gestion spéciale pour left/right TMJ
        if 'left TMJ' in valeurs_trouvees and 'right TMJ' in valeurs_trouvees:
            if 'left and right TMJ' in options_autorisees:
                valeurs_trouvees = [v for v in valeurs_trouvees if v not in ['left TMJ', 'right TMJ']]
                valeurs_trouvees.insert(0, 'left and right TMJ')
        
        # Si "left and right TMJ" existe déjà, supprimer "left TMJ" et "right TMJ" individuels
        if 'left and right TMJ' in valeurs_trouvees:
            valeurs_trouvees = [v for v in valeurs_trouvees if v not in ['left TMJ', 'right TMJ']]
        
        # Éliminer les doublons tout en préservant l'ordre
        valeurs_uniques = []
        for v in valeurs_trouvees:
            if v not in valeurs_uniques:
                valeurs_uniques.append(v)
        valeurs_trouvees = valeurs_uniques
        
        if valeurs_trouvees:
            return ' | '.join(valeurs_trouvees)
    
    # Détection spéciale PRIORITAIRE pour motor vehicle accident (avant la boucle générale)
    if 'motor vehicle accident' in options_autorisees:
        keywords = ['car', 'accident', 'motorcycle', 'vehicle', 'motorbike', 'mva', 'auto']
        if any(keyword in valeur_nettoyee for keyword in keywords):
            return 'motor vehicle accident'
    
    # Pour les indicateurs à valeur unique, on cherche la première correspondance
    for option in options_autorisees:
        option_nettoyee = option.lower()
        
        # Ignorer les options génériques pour la correspondance floue
        if option in ['whatever', 'unknown', 'none']:
            continue
        
        # L'option est contenue dans la valeur
        if option_nettoyee in valeur_nettoyee:
            return option
        
        # La valeur est contenue dans l'option (seulement si > 3 caractères)
        if len(valeur_nettoyee) > 3 and valeur_nettoyee in option_nettoyee:
            return option
    
    # === CORRESPONDANCE PAR MOTS ===
    mots_valeur = set(valeur_nettoyee.split())
    for option in options_autorisees:
        if option in ['whatever', 'unknown', 'none']:
            continue
        mots_option = set(option.lower().split())
        # S'il y a une intersection de mots
        if mots_valeur & mots_option:
            return option
    
    # === CAS FINAL : "whatever" ou "unknown" ===
    # Si la valeur est explicitement "unknown", retourner "unknown"
    if valeur_nettoyee == 'unknown':
        return 'unknown'
    
    # Si "whatever" est disponible, retourner la valeur originale telle quelle
    if has_whatever:
        return valeur_originale.strip()
    
    # Sinon, retourner "unknown"
    return 'unknown'


def harmoniser_summary(texte_summary: str) -> Dict[str, str]:
    """
    Harmoniser un summary en parsant chaque ligne et en normalisant les valeurs
    """
    lignes = texte_summary.strip().split('\n')
    resultat = {}
    
    for ligne in lignes:
        if ':' not in ligne:
            continue
        
        parties = ligne.split(':', 1)
        if len(parties) != 2:
            continue
        
        cle = parties[0].strip()
        valeur = parties[1].strip()
        
        if cle in indicators:
            options_autorisees = indicators[cle]
            valeur_normalisee = normaliser_valeur(valeur, options_autorisees, cle)
            resultat[cle] = valeur_normalisee
        else:
            resultat[cle] = valeur
    
    return resultat


def traiter_fichiers(limite: int = None, aleatoire: bool = False):
    """
    Traiter tous les fichiers de summary avec une limite optionnelle
    """
    import random
    
    fichiers = sorted([f for f in os.listdir(input_dir) if f.endswith('_pred.txt')])
    
    if limite:
        if aleatoire:
            fichiers = random.sample(fichiers, min(limite, len(fichiers)))
            fichiers = sorted(fichiers)  # Trier pour affichage cohérent
        else:
            fichiers = fichiers[:limite]
    
    statistiques = {
        'total_fichiers': len(fichiers),
        'total_unknown': 0,
        'total_indicateurs': len(indicators),
        'fichiers_traites': 0
    }
    
    print(f"\n{'='*70}")
    print(f"Démarrage de l'harmonisation ({len(fichiers)} fichiers)")
    print(f"{'='*70}\n")
    
    for nom_fichier in fichiers:
        chemin_entree = os.path.join(input_dir, nom_fichier)
        
        try:
            with open(chemin_entree, 'r', encoding='utf-8') as f:
                texte_summary = f.read()
            
            # Harmoniser
            resultat = harmoniser_summary(texte_summary)
            
            # Compter les "unknown"
            nb_unknown = sum(1 for v in resultat.values() if v == 'unknown')
            statistiques['total_unknown'] += nb_unknown
            statistiques['fichiers_traites'] += 1
            
            # Formater la sortie
            lignes_sortie = [f"{cle}: {valeur}" for cle, valeur in resultat.items()]
            texte_sortie = '\n'.join(lignes_sortie)
            
            # Sauvegarder
            chemin_sortie = os.path.join(output_dir, nom_fichier)
            with open(chemin_sortie, 'w', encoding='utf-8') as f:
                f.write(texte_sortie)
            
            pourcentage = (nb_unknown / len(resultat)) * 100 if resultat else 0
            print(f"✓ {nom_fichier:30} - {nb_unknown}/{len(resultat)} unknown ({pourcentage:.1f}%)")
            
        except Exception as e:
            print(f"✗ Erreur avec {nom_fichier}: {e}")
            continue
    
    # Afficher le résumé
    print(f"\n{'='*70}")
    print(f"HARMONISATION TERMINÉE")
    print(f"{'='*70}")
    print(f"Fichiers traités:           {statistiques['fichiers_traites']}/{statistiques['total_fichiers']}")
    print(f"Total valeurs 'unknown':    {statistiques['total_unknown']}")
    if statistiques['fichiers_traites'] > 0:
        moyenne = statistiques['total_unknown'] / statistiques['fichiers_traites']
        print(f"Moyenne 'unknown' / fichier: {moyenne:.2f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys
    
    # Par défaut : traiter 5 fichiers aléatoires
    # Pour traiter tous les fichiers : python3 harmonize.py --all
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        print("MODE COMPLET - traitement de TOUS les fichiers\n")
        traiter_fichiers()
    else:
        print("MODE TEST - traitement de 5 fichiers aléatoires")
        print("(Pour traiter tous les fichiers : python3 harmonize.py --all)\n")
        traiter_fichiers(limite=5, aleatoire=True)
