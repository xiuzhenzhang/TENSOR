import re
from collections import defaultdict

datasets = ["Egypt", "Russia_1", "China_1", "Iran_1", "UAE"]
base_path = "useful_utilities/io_dataset/results/"
file_name = "AP_openai_gpt-oss-120b_BAAI_bge-reranker-v2-m3_test_nll.pkl.bz2.txt"

all_scores = defaultdict(dict)

for ds in datasets:
    path = f"{base_path}{ds}/{file_name}"
    try:
        with open(path, 'r') as f:
            content = f.read()

        # Find all temperature blocks
        # Format: For temperature {temp}\nAverage Precision Score w/ LLM: {score}±{std}
        matches = re.findall(r"For temperature ([\d\.]+)\nAverage Precision Score w/ LLM: ([\d\.]+)", content)
        for temp_str, score_str in matches:
            temp = float(temp_str)
            score = float(score_str)
            # Round temperature to 2 decimal places to handle floating point issues
            temp = round(temp, 2)
            all_scores[ds][temp] = score
    except FileNotFoundError:
        print(f"File not found: {path}")

# Find common temperatures across all datasets
common_temps = set(all_scores[datasets[0]].keys())
for ds in datasets[1:]:
    common_temps &= set(all_scores[ds].keys())

common_temps = sorted(list(common_temps))

# Rank temperatures for each dataset
rankings = defaultdict(dict)
for ds in datasets:
    # Sort temperatures by score descending
    sorted_temps = sorted(common_temps, key=lambda t: all_scores[ds][t], reverse=True)
    for rank, temp in enumerate(sorted_temps, 1):
        rankings[ds][temp] = rank

# Calculate average ranking
avg_rankings = {}
for temp in common_temps:
    total_rank = sum(rankings[ds][temp] for ds in datasets)
    avg_rankings[temp] = total_rank / len(datasets)

# Find winning temperature (lowest average rank)
winning_temp = min(avg_rankings, key=avg_rankings.get)

# Output results
output_path = "useful_utilities/io_dataset/temperature_rankings_gpt_oss.txt"
with open(output_path, 'w') as f:
    f.write("Temperature Rankings (Average Rank across datasets):\n")
    for temp in sorted(avg_rankings.keys()):
        f.write(f"Temperature {temp}: {avg_rankings[temp]:.2f}\n")
    f.write(f"\nWinning Temperature: {winning_temp}\n")

print(f"Results written to {output_path}")
