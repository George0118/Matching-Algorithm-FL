import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Function to parse accuracies and losses from log files
def parse_log_file(file_path):
    accuracies = {'RCI_URBAN': {}, 'RCI_SUBURBAN': {}, 'RCI_RURAL': {}, 'RII_URBAN': {}, 'RII_SUBURBAN': {}, 'RII_RURAL': {}}
    losses = {'RCI_URBAN': {}, 'RCI_SUBURBAN': {}, 'RCI_RURAL': {}, 'RII_URBAN': {}, 'RII_SUBURBAN': {}, 'RII_RURAL': {}}

    with open(file_path, 'r') as file:
        for line in file:
            if 'Matching:' in line:
                matching = line.split(":")[1].strip().split(",")[0]

            if matching.startswith("GT"):
                continue

            if 'Fire Server' in line or 'Flood Server' in line or 'Earthquake Server' in line:
                current_server = re.search(r'Fire Server|Flood Server|Earthquake Server', line).group()
                loss_values = [float(x) for x in re.findall(r'Losses: \[([\d.,\s]+)\]', next(file))[0].split(', ')]
                accuracy_values = [float(x) for x in re.findall(r'Accuracies: \[([\d.,\s]+)\]', next(file))[0].split(', ')]
                losses[matching][current_server] = loss_values[-1]  # Get the last loss value
                accuracies[matching][current_server] = accuracy_values[-1]  # Get the last accuracy value

    return accuracies, losses

# Function to plot data for each server
def plot_averages(data, title, save_dir=None):
    plt.figure(figsize=(10, 6))
    for server, server_data in data.items():
        num_users = list(server_data.keys())
        average_accuracies = [user_data['average_accuracy'] for user_data in server_data.values() if 'average_accuracy' in user_data]
        average_losses = [user_data['average_loss'] for user_data in server_data.values() if 'average_loss' in user_data]

        if average_accuracies:
            plt.plot(num_users, average_accuracies, label=f"{server} (Avg Accuracy)", color=colors[server])
        if average_losses:
            plt.plot(num_users, average_losses, label=f"{server} (Avg Loss)", color=colors[server])
                
    plt.title(title)
    plt.xlabel('Number of Users')
    plt.ylabel('Accuracy' if 'Accuracies' in title else 'Loss')
    plt.legend()
    plt.grid(True)
    
    # Save plot if save_dir is provided
    if save_dir:
        filename = title.replace(' ', '_') + '.png'
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        print(f"Plot saved to: {save_path}")
    else:
        plt.show()


# Directory containing log files
directory = '../../results/regret_matching_FL'
save_directory = './fl-areas/regret'
os.makedirs(save_directory, exist_ok=True)

scalability_accuracies = {'RCI_URBAN': {}, 'RCI_SUBURBAN': {}, 'RCI_RURAL': {}, 'RII_URBAN': {}, 'RII_SUBURBAN': {}, 'RII_RURAL': {}}
scalability_losses = {'RCI_URBAN': {}, 'RCI_SUBURBAN': {}, 'RCI_RURAL': {}, 'RII_URBAN': {}, 'RII_SUBURBAN': {}, 'RII_RURAL': {}}

colors = {'Fire Server': 'r', 'Flood Server': 'b', 'Earthquake Server': 'g'}

# Iterate over files in directory
for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory, filename)
        accuracies, losses = parse_log_file(file_path)

        pattern = r'([a-zA-Z]+)-u(\d+)_'
        match = re.search(pattern, filename)
        num_users = match.group(2)

        for matching, data in accuracies.items():
            for server, accuracy in data.items():
                if server not in scalability_accuracies[matching]:
                    scalability_accuracies[matching][server] = {'accuracies': [], 'users': []}
                    scalability_losses[matching][server] = {'losses': [], 'users': []}

                scalability_accuracies[matching][server]['accuracies'].append(data[server])
                scalability_accuracies[matching][server]['users'].append(num_users) 
                scalability_losses[matching][server]['losses'].append(losses[matching][server])
                scalability_losses[matching][server]['users'].append(num_users) 


# Initialize dictionaries to store cumulative totals and counts for accuracies and losses
average_accuracies = {'RCI_URBAN': {}, 'RCI_SUBURBAN': {}, 'RCI_RURAL': {}, 'RII_URBAN': {}, 'RII_SUBURBAN': {}, 'RII_RURAL': {}}
average_losses = {'RCI_URBAN': {}, 'RCI_SUBURBAN': {}, 'RCI_RURAL': {}, 'RII_URBAN': {}, 'RII_SUBURBAN': {}, 'RII_RURAL': {}}

# Function to calculate averages
def calculate_averages(data, averages, str):
    for area, area_data in data.items():
        for server, server_data in area_data.items():
            for value, num_users in zip(server_data[str], server_data['users']):
                if server not in averages[area]:
                    averages[area][server] = {}
                if num_users not in averages[area][server]:
                    averages[area][server][num_users] = {'total': 0, 'count': 0}
                averages[area][server][num_users]['total'] += value
                averages[area][server][num_users]['count'] += 1

# Calculate average accuracies
calculate_averages(scalability_accuracies, average_accuracies, 'accuracies')

# Calculate average losses
calculate_averages(scalability_losses, average_losses, 'losses')

# Calculate the average accuracies and losses for each num_users and server
for area, area_data in average_accuracies.items():
    for server, server_data in area_data.items():
        for num_users, accuracy_stats in server_data.items():
            # Calculate the average accuracy and loss
            average_accuracy = accuracy_stats['total'] / accuracy_stats['count']
            average_loss = average_losses[area][server][num_users]['total'] / average_losses[area][server][num_users]['count']
            # Store the averages in a new key in the averages dictionaries
            average_accuracies[area][server][num_users]['average_accuracy'] = average_accuracy
            average_losses[area][server][num_users]['average_loss'] = average_loss

users = ["12", "15", "18", "21", "24", "27", "30"]

# Summing average accuracy for each server
sum_accuracy = {'RCI_URBAN': [0] * len(users), 'RCI_SUBURBAN': [0] * len(users), 'RCI_RURAL': [0] * len(users), 'RII_URBAN': [0] * len(users), 'RII_SUBURBAN': [0] * len(users), 'RII_RURAL': [0] * len(users)}
sum_loss = {'RCI_URBAN': [0] * len(users), 'RCI_SUBURBAN': [0] * len(users), 'RCI_RURAL': [0] * len(users), 'RII_URBAN': [0] * len(users), 'RII_SUBURBAN': [0] * len(users), 'RII_RURAL': [0] * len(users)}

for area, area_data in average_accuracies.items():
    for server, user_data in area_data.items():
        for user_count, metrics in user_data.items():
            sum_accuracy[area][int((int(user_count)-12)/3)] += metrics['average_accuracy']

for area, area_data in average_losses.items():
    for server, user_data in area_data.items():
        for user_count, metrics in user_data.items():
            sum_loss[area][int((int(user_count)-12)/3)] += metrics['average_loss']


# Plotting
plt.figure(figsize=(10, 8))
for area, values in sum_accuracy.items():
    mean_values = np.mean(values)
    plt.bar(area, mean_values)

plt.title('Average Accuracy for Different Areas', fontsize = 18)
plt.xlabel('Areas', fontsize = 18)
plt.ylabel('Average Server Accuracy', fontsize = 18)
plt.legend(fontsize = 16)
plt.xticks(fontsize = 16, rotation = 45) 
plt.yticks(fontsize = 16)
plt.ylim(bottom = 2.65, top = 2.8)
plt.grid(True)
plt.subplots_adjust(bottom=0.25)
# Save plot if save_dir is provided
if save_directory:
    filename = 'Area_Average_Accuracy.png'
    save_path = os.path.join(save_directory, filename)
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
else:
    plt.show()

plt.figure(figsize=(10, 8))
for area, values in sum_loss.items():
    mean_values = np.mean(values)
    plt.bar(area, mean_values)

plt.title('Average Loss for Different Areas', fontsize = 18)
plt.xlabel('Areas', fontsize = 18)
plt.ylabel('Average Loss', fontsize = 18)
plt.xticks(fontsize = 16, rotation = 45) 
plt.yticks(fontsize = 16)
plt.ylim(bottom = 0.6, top = 1)
plt.legend(fontsize = 16)
plt.subplots_adjust(bottom=0.25)
plt.grid(True)
# Save plot if save_dir is provided
if save_directory:
    filename = 'Area_of_Average_Loss.png'
    save_path = os.path.join(save_directory, filename)
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
else:
    plt.show()

