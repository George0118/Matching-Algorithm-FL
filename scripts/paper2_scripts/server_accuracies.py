import re
import matplotlib.pyplot as plt
import numpy as np
import os

# Directory with your files
dir_path = "../../../results/regret_matching_FL"

servers = ["Fire", "Flood", "Earthquake"]
categories = ["GT", "RCI", "RII"]

# Store a list of accuracy lists for each category+server
accuracy_data_all = {f"{cat}_{srv}": [] for srv in servers for cat in categories}

# Iterate over all .txt files in the directory
for filename in os.listdir(dir_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(dir_path, filename)
        accuracy_data = {f"{cat}_{srv}": [] for srv in servers for cat in categories}
        current_category = None

        with open(file_path, "r") as f:
            for line in f:
                # Detect category
                cat_match = re.search(r"\b(GT|RCI|RII)_URBAN\b", line)
                if cat_match:
                    current_category = cat_match.group(1)

                # Detect server
                srv_match = re.search(r"(Fire|Flood|Earthquake) Server:", line)
                if srv_match:
                    current_server = srv_match.group(1)
                    for subline in f:
                        acc_match = re.search(r"Accuracies:\s*\[([^\]]+)\]", subline)
                        if acc_match:
                            accuracies = [float(x.strip()) for x in acc_match.group(1).split(",")]
                            if current_category:
                                accuracy_data[f"{current_category}_{current_server}"] = accuracies
                            break
        
        # Add the data from this file into the "all" accumulator
        for key, values in accuracy_data.items():
            if values:
                accuracy_data_all[key].append(values)

# Compute averages
average_data = {}
for key, list_of_lists in accuracy_data_all.items():
    if list_of_lists:
        average_data[key] = np.mean(np.array(list_of_lists), axis=0)

# Debug check
for k, v in average_data.items():
    print(k, len(v), "averaged values")

# Create output directory if it doesn't exist
output_dir = "accuracies"
os.makedirs(output_dir, exist_ok=True)

# Plot
plt.figure(figsize=(12, 6))
for key, values in average_data.items():
    plt.plot(range(1, len(values) + 1), values, label=key)

plt.title("Average Server Accuracies across 80 Epochs (URBAN)")
plt.xlabel("Epoch")
plt.ylabel("Average Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save plot as PNG
output_path = os.path.join(output_dir, "average_server_accuracies.png")
plt.savefig(output_path, dpi=300)

plt.show()
print(f"Plot saved to: {output_path}")
