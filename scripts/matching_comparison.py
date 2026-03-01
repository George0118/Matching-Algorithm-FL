import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

color_mapping_1 = {
    "fire": "red",
    "flood": "blue",
    "earthquake": "green"
}

color_mapping_2 = {
    "GT": "blue",
    "RCI": "orange",
    "RII": "green"
}

color_mapping_3 = {
    "URBAN": "blue",
    "SUBURBAN": "orange",
    "RURAL": "green"
}

results_directory = "../../results/regret_matching"
matching_data = {}

# New data structure for storing area-specific data for RCI and RII
area_data = {"RCI": {},
             "RII": {}}

def extract_value(line):
    match = re.search(r'\b\d+\.\d+\b|\b\d+\b', line)
    if match:
        value = match.group()
        return float(value) if '.' in value else int(value)
    return None

# Loop through files in the results directory
for filename in os.listdir(results_directory):
    file_path = os.path.join(results_directory, filename)

    if os.path.isfile(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

            matching = users = area = None
            for line in lines:
                match = re.match(r'Matching: (\w+_\w+), Users: (\d+), Critical Points: (\d+)', line)
                if match:
                    matching_full, users = match.groups()[0], int(match.groups()[1])
                    area_match = re.search(r'_(URBAN|SUBURBAN|RURAL)$', matching_full)
                    if area_match:
                        area = area_match.group(1)
                    matching = re.sub(r'(_URBAN|_SUBURBAN|_RURAL)$', '', matching_full)
                    matching_data.setdefault(matching, {})

                elif "Mean Energy" in line or "Mean Elocal" in line or "Mean Etransfer" in line \
                        or "Mean Datarate" in line or "Mean User Utility" in line or "Mean Server Utility" in line \
                        or "Sum User Payments" in line or "Time" in line or "Iterations" in line\
                        or "Mean Energy" in line:
                    magnitude_match = re.search(r'^\s*([^:]+)\s*:', line)
                    if magnitude_match:
                        magnitude = magnitude_match.group(1).strip()
                        if matching == "GT" and magnitude == "Iterations":
                            continue
                        value = extract_value(line)
                        matching_data[matching].setdefault(magnitude, {"Users": [], "Values": []})
                        matching_data[matching][magnitude]["Users"].append(users)
                        matching_data[matching][magnitude]["Values"].append(value)

                        # Store Time and Iterations data for area-specific RCI and RII matchings
                        if magnitude in ["Time", "Iterations"] and matching in ["RCI", "RII"] and area:
                            area_data[matching].setdefault(magnitude, {"URBAN": {"Users": [], "Values": []}, "SUBURBAN": {"Users": [], "Values": []}, "RURAL": {"Users": [], "Values": []}})
                            area_data[matching][magnitude][area]["Users"].append(users)
                            area_data[matching][magnitude][area]["Values"].append(value)

# Existing plotting code for each magnitude
user_values = [12, 15, 18, 21, 24, 27, 30]
save_directory = "./matching_comparison/regret_comparison"
os.makedirs(save_directory, exist_ok=True)
magnitudes = ["Mean Energy", "Mean Elocal", "Mean Etransfer", "Mean Datarate", "Mean User Utility", "Mean Server Utility", "Time", "Iterations"]

for magnitude in magnitudes:
    plt.figure(figsize=(10, 6))
    plt.xlabel("Nodes", fontsize=18)
    if magnitude == "Mean User Utility":
        plt.ylabel("Mean Node Utility", fontsize=18)
    else:
        plt.ylabel(magnitude, fontsize=18)

    for matching, data in matching_data.items():

        if matching == "GT" and magnitude == "Iterations":
            continue

        users = data[magnitude]["Users"]
        values = data[magnitude]["Values"]
        
        # Step 1: Group the values 
        grouped_data = defaultdict(list)

        for user, val in zip(users, values):
            grouped_data[user].append(val)

        # Step 2: Calculate the mean for each group
        average_values = {user: np.mean(vals) for user, vals in grouped_data.items()}

        if matching == "GT":
            m = "Game Theory"
        elif matching == "RCI":
            m = "Regret Complete Information"
        else:
            m = "Regret Incomplete Information"

        users_sorted = sorted(average_values.keys())  # Sort magnitudes

        if magnitude == "Time":
            averages_sorted = [average_values[user]/user for user in users_sorted]
        else:
            averages_sorted = [average_values[user] for user in users_sorted]

        plt.plot(users_sorted, averages_sorted, marker='o', label=f"Matching: {m}", c = color_mapping_2[matching])

    plt.xticks(user_values, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    
    # Save the plot as PNG
    plt.savefig(os.path.join(save_directory, f"{magnitude.replace(' ', '_')}_vs_Users.png"), bbox_inches='tight')

# Existing bar plots for averages
for magnitude in magnitudes:
    plt.figure(figsize=(10, 6))
    if magnitude == "Mean User Utility":
        plt.title(f"Average Mean Node Utility for Each Matching", fontsize = 18)
    else:
        plt.title(f"Average {magnitude} for Each Matching", fontsize = 18)
    plt.xlabel("Matching", fontsize = 18)
    if magnitude == "Mean User Utility":
        plt.ylabel("Average Mean Node Utility", fontsize = 18)
    else:
        plt.ylabel("Average " + magnitude, fontsize = 18)

    avg_values = []
    algorithms = ["Game Theory", "Regret Complete Information", "Regret Incomplete Information"]

    for matching, data in matching_data.items():
        if matching == "GT" and magnitude == "Iterations":
            algorithms = ["Regret Complete Information", "Regret Incomplete Information"]
            continue
        values = data[magnitude]["Values"]
        avg_value = np.mean(values)
        avg_values.append(avg_value)

    colors = [color_mapping_2[value] for value in ["GT", "RCI", "RII"]]
    bars = plt.bar(algorithms, avg_values, color=colors, width=0.3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    for bar, value in zip(bars, avg_values):
        if magnitude != "Mean Datarate":
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01*bar.get_height(), f'{value:.7f}', ha='center', va='bottom')
        else:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01*bar.get_height(), f'{value:.2f}', ha='center', va='bottom')
            
    algorithms = ["Game Theory", "Regret Complete Information", "Regret Incomplete Information"]

    plt.savefig(os.path.join(save_directory, f"Average_{magnitude.replace(' ', '_')}.png"), bbox_inches='tight')


areas = ["URBAN", "SUBURBAN", "RURAL"]
magnitudes = ["Time", "Iterations"]
matchings = {"RCI": "Regret Complete Information", "RII": "Regret Incomplete Information"}

# New: Bar plots for average Time and Iterations for each area in RCI and RII
for magnitude in magnitudes:
    for matching_key, matching_name in matchings.items():

        plt.figure(figsize=(10, 6))
        plt.title(f"Average {magnitude} for {matching_name}", fontsize = 18)
        plt.xlabel("Matching", fontsize = 18)
        plt.ylabel(f"Average {magnitude}", fontsize = 18)

        avg_values = []

        for area in areas:
            if magnitude == "Time":
                values = [v / u for v, u in zip(area_data[matching_key][magnitude][area]["Values"], area_data[matching_key][magnitude][area]["Users"])]
            else:
                values = area_data[matching_key][magnitude][area]["Values"]

            avg_value = np.mean(values)
            avg_values.append(avg_value)

        colors = [color_mapping_2[value] for value in ["GT", "RCI", "RII"]]
        bars = plt.bar(areas, avg_values, color=colors, width=0.3)
        
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)

        for bar, value in zip(bars, avg_values):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01*bar.get_height(), f'{value:.2f}', ha='center', va='bottom')

        plt.savefig(os.path.join(save_directory, f"Average_{magnitude}_per_area_{matching_key}.png"), bbox_inches='tight')


for matching_key, matching_name in matchings.items():
    plt.figure(figsize=(10, 6))
    plt.title(f"Average Iterations for different number of Nodes for {matching_name}", fontsize = 18)
    plt.xlabel("Nodes", fontsize = 18)
    plt.ylabel(f"Average Iterations", fontsize = 18)

    for area in areas:
        values = area_data[matching_key]["Iterations"][area]["Values"]
        users = area_data[matching_key]["Iterations"][area]["Users"]

        # Step 1: Group the values 
        grouped_data = defaultdict(list)

        for user, val in zip(users, values):
            grouped_data[user].append(val)

        # Step 2: Calculate the mean for each group
        average_values = {user: np.mean(vals) for user, vals in grouped_data.items()}

        users_sorted = sorted(average_values.keys())  # Sort magnitudes
        averages_sorted = [average_values[user] for user in users_sorted]

        plt.plot(users_sorted, averages_sorted, marker='o', label=f"Area: {area}", c = color_mapping_3[area])

    plt.xticks(user_values, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    
    # Save the plot as PNG
    plt.savefig(os.path.join(save_directory, f"{matching_key}_Iterations_vs_Users_per_Area.png"), bbox_inches='tight')

# Time vs Users per Area plots
for matching_key, matching_name in matchings.items():
    plt.figure(figsize=(10, 6))
    plt.title(f"Average Time for different number of Nodes for {matching_name}", fontsize = 18)
    plt.xlabel("Nodes", fontsize = 18)
    plt.ylabel(f"Average Time", fontsize = 18)

    for area in areas:
        values = area_data[matching_key]["Time"][area]["Values"]
        users = area_data[matching_key]["Time"][area]["Users"]

        # Step 1: Group the values 
        grouped_data = defaultdict(list)

        for user, val in zip(users, values):
            grouped_data[user].append(val)

        # Step 2: Calculate the mean for each group
        average_values = {user: np.mean(vals) for user, vals in grouped_data.items()}

        users_sorted = sorted(average_values.keys())  # Sort magnitudes
        averages_sorted = [average_values[user]/user for user in users_sorted]

        plt.plot(users_sorted, averages_sorted, marker='o', label=f"Area: {area}", c = color_mapping_3[area])

    plt.xticks(user_values, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    
    # Save the plot as PNG
    plt.savefig(os.path.join(save_directory, f"{matching_key}_Time_vs_Users_per_Area.png"), bbox_inches='tight')