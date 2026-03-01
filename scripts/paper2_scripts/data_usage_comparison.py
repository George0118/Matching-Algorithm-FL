import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Color mapping for different algorithms
color_mapping = {
    "GT": "blue",
    "RCI": "orange", 
    "RII": "green"
}

# Algorithm name mapping
algorithm_names = {
    "GT": "Game Theory",
    "RCI": "Regret Complete Information",
    "RII": "Regret Incomplete Information"
}

def extract_user_count(filename):
    """Extract user count from filename like 'Areas-u12_cp3_08-04_14-36-17_2'"""
    match = re.search(r'u(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def extract_value(line):
    """Extract numeric value from a line"""
    match = re.search(r'\b\d+\.\d+\b|\b\d+\b', line)
    if match:
        value = match.group()
        return float(value) if '.' in value else int(value)
    return None

def parse_file(file_path):
    """Parse a single results file and extract data usage information"""
    data = {}
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        current_matching = None
        current_users = None
        
        for line in lines:
            # Extract matching algorithm and user count
            match = re.match(r'Matching: (\w+_\w+), Users: (\d+), Critical Points: (\d+)', line)
            if match:
                matching_full, users = match.groups()[0], int(match.groups()[1])
                current_users = users
                
                # Extract algorithm name (GT, RCI, RII)
                algorithm_match = re.match(r'(\w+)_', matching_full)
                if algorithm_match:
                    current_matching = algorithm_match.group(1)
                    data[current_matching] = {
                        'users': current_users,
                        'percentage_data_used': None
                    }
            
            # Extract Percentage of Data Used
            elif "Percentage of Data Used" in line:
                if current_matching:
                    value = extract_value(line)
                    if value is not None:
                        data[current_matching]['percentage_data_used'] = value
    
    return data

def main():
    # Directory containing the results files
    results_directory = "../../../results/regret_matching"
    
    # Dictionary to store data organized by user count
    user_data = defaultdict(lambda: defaultdict(list))
    
    # Process all files in the results directory
    for filename in os.listdir(results_directory):
        file_path = os.path.join(results_directory, filename)
            
        if os.path.isfile(file_path):
            user_count = extract_user_count(filename)
            if user_count:
                
                file_data = parse_file(file_path)
                
                # Store data for each algorithm
                for algorithm, data in file_data.items():
                    if data['percentage_data_used'] is not None:
                        user_data[user_count][algorithm].append(data['percentage_data_used'])

    
    # Create plots
    save_directory = "./data_usage_comparison"
    os.makedirs(save_directory, exist_ok=True)
    
    # Get all user counts and sort them
    user_counts = sorted(user_data.keys())
    
    # Plot 1: Line plot showing data usage vs user count for each algorithm
    plt.figure(figsize=(12, 8))
    plt.title("Percentage of Data Used vs Number of Users", fontsize=18)
    plt.xlabel("Number of Users", fontsize=16)
    plt.ylabel("Percentage of Data Used", fontsize=16)
    
    for algorithm in ["GT", "RCI", "RII"]:
        if algorithm in algorithm_names:
            values = []
            for user_count in user_counts:
                if algorithm in user_data[user_count]:
                    avg_value = np.mean(user_data[user_count][algorithm])
                    values.append(avg_value)
                else:
                    values.append(None)
            
            # Filter out None values
            valid_indices = [i for i, v in enumerate(values) if v is not None]
            valid_user_counts = [user_counts[i] for i in valid_indices]
            valid_values = [values[i] for i in valid_indices]
            
            if valid_values:
                plt.plot(valid_user_counts, valid_values, marker='o', 
                        label=algorithm_names[algorithm], 
                        color=color_mapping[algorithm], 
                        linewidth=2, markersize=8)
    
    plt.xticks(user_counts, fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "data_usage_vs_users.png"), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot 2: Bar plot comparing average data usage across algorithms
    plt.figure(figsize=(10, 6))
    plt.title("Average Percentage of Data Used by Algorithm", fontsize=18)
    plt.xlabel("Algorithm", fontsize=16)
    plt.ylabel("Average Percentage of Data Used", fontsize=16)
    
    algorithms = ["GT", "RCI", "RII"]
    avg_values = []
    
    for algorithm in algorithms:
        all_values = []
        for user_count in user_data:
            if algorithm in user_data[user_count]:
                all_values.extend(user_data[user_count][algorithm])
        
        if all_values:
            avg_values.append(np.mean(all_values))
        else:
            avg_values.append(0)
    
    bars = plt.bar([algorithm_names[alg] for alg in algorithms], avg_values, 
                   color=[color_mapping[alg] for alg in algorithms], alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontsize=12)
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "average_data_usage_by_algorithm.png"), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot 3: Heatmap showing data usage for each user count and algorithm
    plt.figure(figsize=(12, 8))
    
    # Prepare data for heatmap
    heatmap_data = []
    for algorithm in algorithms:
        row = []
        for user_count in user_counts:
            if algorithm in user_data[user_count]:
                avg_value = np.mean(user_data[user_count][algorithm])
                row.append(avg_value)
            else:
                row.append(np.nan)
        heatmap_data.append(row)
    
    # Create heatmap
    im = plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    # Set labels
    plt.xticks(range(len(user_counts)), user_counts)
    plt.yticks(range(len(algorithms)), [algorithm_names[alg] for alg in algorithms])
    plt.xlabel("Number of Users", fontsize=16)
    plt.ylabel("Algorithm", fontsize=16)
    plt.title("Percentage of Data Used Heatmap", fontsize=18)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Percentage of Data Used", fontsize=14)
    
    # Add text annotations
    for i in range(len(algorithms)):
        for j in range(len(user_counts)):
            if not np.isnan(heatmap_data[i][j]):
                plt.text(j, i, f'{heatmap_data[i][j]:.3f}', 
                        ha='center', va='center', fontsize=10, color='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, "data_usage_heatmap.png"), 
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print summary statistics
    print("\n=== DATA USAGE SUMMARY ===")
    print(f"Processed {len(user_counts)} different user counts: {user_counts}")
    
    for algorithm in algorithms:
        all_values = []
        for user_count in user_data:
            if algorithm in user_data[user_count]:
                all_values.extend(user_data[user_count][algorithm])
        
        if all_values:
            print(f"\n{algorithm_names[algorithm]}:")
            print(f"  Average: {np.mean(all_values):.4f}")
            print(f"  Std Dev: {np.std(all_values):.4f}")
            print(f"  Min: {np.min(all_values):.4f}")
            print(f"  Max: {np.max(all_values):.4f}")
            print(f"  Count: {len(all_values)}")
    
    print(f"\nPlots saved in: {save_directory}")

if __name__ == "__main__":
    main() 