import os
import json
from rouge_score import rouge_scorer, scoring

# Paths
manual_folder = "/home/luciacev/Desktop/LLM/data_predicition/data_max_harmonized"
llm_folder = "/home/luciacev/Desktop/LLM/Mistral-7B-instruct/predict_Mistral7B_eval"

if not os.path.exists(manual_folder) or not os.path.exists(llm_folder):
    print("Error: Folders not found.")
    exit(1)

print("=" * 80)
print("CALCULATING ROUGE SCORES - WITH AND WITHOUT UNKNOWN VALUES")
print("=" * 80)

def remove_unknown_lines(text):
    """Remove lines containing 'unknown' from text."""
    lines = [l for l in text.split('\n') if l.strip() and 'unknown' not in l.lower()]
    return '\n'.join(lines)

# ===== VERSION 1: WITH ALL DATA (including unknown) =====
print("\n[1/2] Computing ROUGE scores WITH all data (including 'unknown')...")
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
aggregator = scoring.BootstrapAggregator()

llm_files = [f for f in os.listdir(llm_folder) if f.endswith('_pred.txt')]
count_with = 0

for llm_file in llm_files:
    patient_id = llm_file.replace('_pred.txt', '')
    manual_file = f"{patient_id}_summary.txt"
    
    manual_path = os.path.join(manual_folder, manual_file)
    llm_path = os.path.join(llm_folder, llm_file)
    
    if os.path.exists(manual_path):
        with open(manual_path, 'r', encoding='utf-8') as f:
            ref = f.read().strip()
        with open(llm_path, 'r', encoding='utf-8') as f:
            pred = f.read().strip()
        
        scores = scorer.score(ref, pred)
        aggregator.add_scores(scores)
        count_with += 1

result = aggregator.aggregate()

rouge_scores_with = {
    'rouge1': result['rouge1'].mid.fmeasure,
    'rouge2': result['rouge2'].mid.fmeasure,
    'rougeL': result['rougeL'].mid.fmeasure,
    'rougeLsum': result['rougeLsum'].mid.fmeasure,
    'num_samples': count_with
}

# ===== VERSION 2: WITHOUT UNKNOWN VALUES =====
print("[2/2] Computing ROUGE scores WITHOUT 'unknown' values...")
scorer2 = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
aggregator2 = scoring.BootstrapAggregator()

count_without = 0

for llm_file in llm_files:
    patient_id = llm_file.replace('_pred.txt', '')
    manual_file = f"{patient_id}_summary.txt"
    
    manual_path = os.path.join(manual_folder, manual_file)
    llm_path = os.path.join(llm_folder, llm_file)
    
    if os.path.exists(manual_path):
        with open(manual_path, 'r', encoding='utf-8') as f:
            ref = f.read().strip()
        with open(llm_path, 'r', encoding='utf-8') as f:
            pred = f.read().strip()
        
        # Remove lines with 'unknown' from both reference and prediction
        ref_clean = remove_unknown_lines(ref)
        pred_clean = remove_unknown_lines(pred)
        
        # Only compute if both have content after filtering
        if ref_clean.strip() and pred_clean.strip():
            scores = scorer2.score(ref_clean, pred_clean)
            aggregator2.add_scores(scores)
            count_without += 1

result2 = aggregator2.aggregate()

rouge_scores_without = {
    'rouge1': result2['rouge1'].mid.fmeasure,
    'rouge2': result2['rouge2'].mid.fmeasure,
    'rougeL': result2['rougeL'].mid.fmeasure,
    'rougeLsum': result2['rougeLsum'].mid.fmeasure,
    'num_samples': count_without
}

# Save both versions
os.makedirs("metrics", exist_ok=True)
with open("metrics/rouge_scores.json", "w") as f:
    json.dump(rouge_scores_with, f, indent=4)

with open("metrics/rouge_scores_no_unknown.json", "w") as f:
    json.dump(rouge_scores_without, f, indent=4)

print("\n✅ Metrics saved:")
print("   - metrics/rouge_scores.json (WITH all data)")
print("   - metrics/rouge_scores_no_unknown.json (WITHOUT 'unknown' lines)")

# Print comparison
print("\n" + "=" * 80)
print("ROUGE COMPARISON: WITH ALL DATA vs WITHOUT 'UNKNOWN' VALUES")
print("=" * 80)

print(f"\n{'Metric':<15} {'WITH all':<18} {'WITHOUT unknown':<18} {'Difference':<15}")
print("-" * 70)

metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
for metric in metrics:
    with_val = rouge_scores_with[metric]
    without_val = rouge_scores_without[metric]
    diff = without_val - with_val
    print(f"{metric:<15} {with_val:<18.4f} {without_val:<18.4f} {diff:+.4f}")

print(f"\n{'Samples':<15} {rouge_scores_with['num_samples']:<18} {rouge_scores_without['num_samples']:<18}")

print("\n" + "=" * 80)
print("INTERPRETATION:")
print("=" * 80)
print("- WITH all: Includes 'unknown' fields in the text comparison")
print("- WITHOUT unknown: Removes all 'unknown' lines before computing ROUGE")
print(f"- Real ROUGE performance (on actual data): {rouge_scores_without['rouge1']:.4f}")
if rouge_scores_without['rouge1'] < rouge_scores_with['rouge1']:
    inflation = rouge_scores_with['rouge1'] - rouge_scores_without['rouge1']
    print(f"- Inflation from 'unknown' fields: {inflation:+.4f}")
else:
    print(f"- Performance improvement when filtering unknown: {rouge_scores_without['rouge1'] - rouge_scores_with['rouge1']:+.4f}")
print("=" * 80 + "\n")

