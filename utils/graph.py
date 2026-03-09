import matplotlib.pyplot as plt
import numpy as np

# --- DONNÉES ---
models = ['Qwen 2.5-7B', 'Mistral 7B', 'Qwen 2.5-1.5B', 'Llama 3-8B', 'DeepSeek', 'Phi-3.5', 'BART']
scores = [0.8992, 0.8964, 0.8845, 0.8805, 0.7141, 0.7089, 0.6881]

# --- DESIGN COMPACT (7x4.5 pouces est idéal pour une colonne) ---
plt.figure(figsize=(7, 4.5), dpi=300) 

# Couleurs : Ton modèle en bleu foncé, les autres en gris clair
colors = ['#00629B' if 'Qwen 2.5' in m else '#00629B' for m in models]

# Création des barres
bars = plt.bar(models, scores, color=colors, width=0.75, edgecolor='black', linewidth=0.6)

# --- AXES & TITRES (Police plus grosse pour la lisibilité) ---
# Note : En IEEE, le vrai titre est dans la légende (caption) SOUS le graph.
# Ici on soigne les labels des axes.
plt.ylabel('Semantic Similarity Score', fontsize=12, fontname='Times New Roman', fontweight='bold')
plt.xlabel('Evaluated Models', fontsize=12, fontname='Times New Roman', fontweight='bold')

# --- CHIFFRES SUR LES BARRES ---
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, 
             f"{yval:.3f}", # Arrondi à 3 chiffres pour gagner de la place
             ha='center', va='bottom', fontsize=10, fontname='Times New Roman', color='black')

# --- NETTOYAGE ---
# Ajustement de l'échelle Y pour laisser de la place aux chiffres
plt.ylim(0, 1.1)

# Rotation des noms pour qu'ils ne se chevauchent pas
plt.xticks(rotation=45, ha='right', fontsize=11, fontname='Times New Roman')
plt.yticks(fontsize=11, fontname='Times New Roman')

# Enlever le cadre du haut et de droite
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Grille légère
plt.grid(axis='y', linestyle='--', alpha=0.4)

# --- COMMANDE MAGIQUE ---
# C'est ça qui empêche que le titre soit coupé ou que l'image soit trop vide
plt.tight_layout()

# --- SAUVEGARDE ---
plt.savefig('benchmark_compact.png', dpi=300)
plt.savefig('benchmark_compact.pdf') # Utilise le PDF pour ton LaTeX si possible
print("Graphs saved: benchmark_compact.png, benchmark_compact.pdf")