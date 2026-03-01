# Project description


# ========================================= #

import random

# Argument Parsing

from config import parse_arguments

parse_arguments()

from config import num
# Affinity

import os
import psutil
from config import thread_count
my_pid = os.getpid()
p = psutil.Process(my_pid)
starting_cpu = num*thread_count
cpu_list = [starting_cpu + i for i in range(thread_count)]
p.cpu_affinity(cpu_list)
print(p.cpu_affinity())

# Imports
import numpy as np
import time
import copy
import math
import datetime
from matchingeq_functions import check_matching_equality
from Data.Classes.Data import Get_data
from Data.load_images import fire_input_paths, flood_input_paths, earthquake_input_paths, count_images
from Data.federated_learning import Servers_FL
from Data.Classes.Model import *
from Data.fl_parameters import *
from Classes.Server import Server
from Classes.User import User, Ptrans_max, fn_max
from Classes.CriticalPoint import CP
from helping_functions import dataset_sizes, h, channel_gain
from Regret_Matching.regret_matching import regret_matching
from Regret_Matching.regret_matching_II import regret_matching_II
from Regret_Matching.utility_function import user_utility_ext
from GT_Matching.approximate_matching import approximate_fedlearner_matching
from GT_Matching.accurate_matching import accurate_fedlearner_matching
from GT_Matching.gt_utilities import server_utility_externality

# General Parameters

from config import N,S,K
from general_parameters import *

urban_threshold = 30    # In an Urban Area, after 30 users the users start to be further in proximity to the CPs
suburban_threshold = 21 # In an Suburban Area, after 21 users the users start to be further in proximity to the CPs
rural_threshold = 12    # In an Rural Area, after 12 users the users start to be further in proximity to the CPs
federated_learning = False

# ===================== Users', Servers' and Critical Points' Topology ===================== #

start_time = time.time()

# Servers: on (0,0,0)
servers = []

while True:
    # Create servers with random p values
    for i in range(S):
        p = math.ceil(0.333 * N)
        server = Server(0,0,0,p,i)
        servers.append(server)

    # Ensure the sum of p is greater than or equal to N
    if sum(server.p for server in servers) >= N:
        break
    else:
        # If the condition is not met, empty the list and try again
        servers = []

for s in servers:
        print("Server", s.num, " Resources: ", s.p)


# Normalize payments
max_server_payment = 0
for s in servers:
    if server.p > max_server_payment:
        max_server_payment = server.p

for s in servers:
    s.set_p(s.p/max_server_payment)
    
# Critical Points: inside a cube centered at (0,0,0) and side = 2
critical_points = []

for i in range(K):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    z = random.uniform(-1, 1)
    distance = math.sqrt(x**2 + y**2 + z**2)

    # Ensure that the distance between two critical points is at least 0.8
    while any(math.sqrt((x - cp.x)**2 + (y - cp.y)**2 + (z - cp.z)**2) < 0.8 for cp in critical_points) or distance < 0.3:
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        z = random.uniform(-1, 1)
        distance = math.sqrt(x**2 + y**2 + z**2)

    disaster = disasters[i % S]

    cp = CP(x, y, z, i, disaster)
    critical_points.append(cp)

# Associate Critical Points with their Servers
for cp in critical_points:
    if(cp.get_disaster() == "fire"):
        servers[0].add_critical_point(cp)       # All fire CPs with Server 1
    elif(cp.get_disaster() == "flood"):
        servers[1].add_critical_point(cp)       # All flood CPs with Server 2
    else:
        servers[2].add_critical_point(cp)       # All earthquake CPs with Server 3

urban_servers = copy.deepcopy(servers)
subruban_servers = copy.deepcopy(servers)
rural_servers = copy.deepcopy(servers)

# Users: on a sphere (radius = 1) around the servers / users with lower i are closer to the CPs 
urban_users = [] 
suburban_users = [] 
rural_users = [] 

distance_diff = 0.005

for i in range(min(rural_threshold, N)):

    while True:
        j = i//K
        cp = critical_points[i%K]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        z = random.uniform(-1,1)

        distance = math.sqrt((cp_x - x)**2 + (cp_y - y)**2 + (cp_z - z)**2)

        if distance < 0.02+(j+1)*distance_diff and distance >= 0.02+j*distance_diff:  # if in the desired sphere and good user then add the user
            break

    user = User(x,y,z,i)
    
    urban_users.append(copy.deepcopy(user))
    suburban_users.append(copy.deepcopy(user))
    rural_users.append(copy.deepcopy(user))

for i in range(min(suburban_threshold - rural_threshold, N - rural_threshold)):

    while True:
        j = i//K
        cp = critical_points[i%K]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        z = random.uniform(-1,1)

        distance = math.sqrt((cp_x - x)**2 + (cp_y - y)**2 + (cp_z - z)**2)

        if distance < 0.02+(rural_threshold+j+1)*distance_diff and distance >= 0.02+(rural_threshold+j)*distance_diff:  # if in the desired sphere and good user then add the user
            break
        
    user = User(x,y,z,i+rural_threshold)
    
    urban_users.append(copy.deepcopy(user))
    suburban_users.append(copy.deepcopy(user))

    while True:
        j = i//K
        cp = critical_points[i%K]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        z = random.uniform(-1,1)

        distance = math.sqrt((cp_x - x)**2 + (cp_y - y)**2 + (cp_z - z)**2)

        if distance < 0.3+(j+1)*distance_diff and distance >= 0.3+j*distance_diff:     # if in the desired sphere and bad user then add the user
            break
        
    user = User(x,y,z,i+rural_threshold)
    
    rural_users.append(copy.deepcopy(user))


for i in range(min(urban_threshold - suburban_threshold, N - suburban_threshold)):

    while True:
        j = i//K
        cp = critical_points[i%K]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        z = random.uniform(-1,1)

        distance = math.sqrt((cp_x - x)**2 + (cp_y - y)**2 + (cp_z - z)**2)

        if distance < 0.02+(suburban_threshold+j+1)*distance_diff and distance >= 0.02+(suburban_threshold+j)*distance_diff:  # if in the desired sphere and good user then add the user
            break
        
    user = User(x,y,z,i+suburban_threshold)
    
    urban_users.append(copy.deepcopy(user))

    while True:
        j = i//K
        cp = critical_points[i%K]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        z = random.uniform(-1,1)

        distance = math.sqrt((cp_x - x)**2 + (cp_y - y)**2 + (cp_z - z)**2)

        if distance < 0.2+(j+1)*distance_diff and distance >= 0.2+j*distance_diff:     # if in the desired sphere and bad user then add the user
            break
        
    user = User(x,y,z,i+suburban_threshold)
    
    suburban_users.append(copy.deepcopy(user))

    while True:
        j = i//K
        cp = critical_points[i%K]
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
        z = random.uniform(-1,1)

        distance = math.sqrt((cp_x - x)**2 + (cp_y - y)**2 + (cp_z - z)**2)

        if distance < 0.3+(suburban_threshold-rural_threshold+j+1)*distance_diff and distance >= 0.3+(suburban_threshold-rural_threshold+j)*distance_diff:     # if in the desired sphere and bad user then add the user
            break
        
    user = User(x,y,z,i+suburban_threshold)
    
    rural_users.append(copy.deepcopy(user))

# ========================================================================================== #  
        
all_users = {'Urban': urban_users, 'Suburban': suburban_users, 'Rural': rural_users}
print()

# ===================== Initialization of Users' Utility Values ===================== #
        
# Data size for each user

total_images = count_images(fire_input_paths + flood_input_paths + earthquake_input_paths)

# For each user calculate the minimum distance from the relevant Critical Points
for area, users in all_users.items():
    Dn = [0]*N
    
    for s in servers:   # For each server(disaster) calculate number of images each user will receive
        cps = s.get_critical_points()   # Get the relevant Critical Points
        sizes = dataset_sizes(s, area, users, cps, total_images, True)

        # And add to the Dn of each user
        for i in range(N):
            Dn[i] += sizes[i]*3*224*224*8 # number of bits for all images

    # Set the User final datasize
    for i in range(N):
        Dn[i] += N_neutral*3*224*224*8  # each user has another 250 neutral images
        user = users[i]
        user.set_datasize(Dn[i])

print()

# =================================================== #

# Set distances

for area, users in all_users.items():
    for i in range(N):
        distances = []
        for j in range(S):          # Finding Max Distance for each user
            user = users[i]
            user_x, user_y, user_z = user.x, user.y, user.z

            s = servers[j]
            server_x, server_y, server_z = s.x, s.y, s.z

            distance = math.sqrt((server_x - user_x)**2 + (server_y - user_y)**2 + (server_z - user_z)**2)

            distances.append(distance)
        user.set_distances(distances)

# ===================================================== #

# =================================================================================== #
    
# Normalized Data Importance

for area, users in all_users.items():
    min_dist = [None]*K
    for i in range(K):
        for j in range(N):          # Finding Min Distance for each CP
            user = users[j]
            user_x, user_y, user_z = user.x, user.y, user.z

            cp = critical_points[i]
            cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

            distance = math.sqrt((cp_x - user_x)**2 + (cp_y - user_y)**2 + (cp_z - user_z)**2)

            if min_dist[i] is None or distance < min_dist[i]:
                min_dist[i] = distance

        for j in range(N):          # Calculating Importance for each user
            user = users[j]
            user_x, user_y, user_z = user.x, user.y, user.z

            cp = critical_points[i]
            cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

            distance = math.sqrt((cp_x - user_x)**2 + (cp_y - user_y)**2 + (cp_z - user_z)**2)   

            importance = min_dist[i] / distance

            user.add_importance(importance)

# ======================================================================================== #


# ========== Calculating parameters for utility functions =========== #

for i in range(N):
    _users = [all_users["Urban"][i], all_users["Suburban"][i], all_users["Rural"][i]]
    for server in servers:
        imp1 = _users[0].get_importance()[server.num]
        imp2 = _users[1].get_importance()[server.num]
        imp3 = _users[2].get_importance()[server.num]
        if i < rural_threshold:
            # Pns
            a_pns_1 = random.uniform(max(0.4*imp1-0.05, 0.05), 0.4*imp1)
            a_pns_2 = a_pns_1
            a_pns_3 = a_pns_1
            # Fn
            a_fn_1 = random.uniform(max(0.57*(math.sqrt(imp1))-0.05, 0.05), 0.57*math.sqrt(imp1))
            a_fn_2 = a_fn_1
            a_fn_3 = a_fn_1
            # Dn
            a_dn_1 = random.uniform(max(0.57*(math.sqrt(imp1))-0.05, 0.05), 0.57*math.sqrt(imp1))
            a_dn_2 = a_dn_1
            a_dn_3 = a_dn_1

        elif i < suburban_threshold:
            # Pns
            a_pns_1 = random.uniform(max(0.4*imp1-0.05, 0.05), 0.4*imp1)
            a_pns_2 = a_pns_1
            a_pns_3 = random.uniform(max(0.4*imp3-0.05, 0.05), 0.4*imp3)
            # Fn
            a_fn_1 = random.uniform(max(0.57*(math.sqrt(imp1))-0.05, 0.05), 0.57*math.sqrt(imp1))
            a_fn_2 = a_fn_1
            a_fn_3 = random.uniform(max(0.57*(math.sqrt(imp3))-0.05, 0.05), 0.57*math.sqrt(imp3))
            # Dn
            a_dn_1 = random.uniform(max(0.57*(math.sqrt(imp1))-0.05, 0.05), 0.57*math.sqrt(imp1))
            a_dn_2 = a_dn_1
            a_dn_3 = random.uniform(max(0.57*(math.sqrt(imp3))-0.05, 0.05), 0.57*math.sqrt(imp3))

        else:
            # Pns
            a_pns_1 = random.uniform(max(0.4*imp1-0.05, 0.05), 0.4*imp1)
            a_pns_2 = random.uniform(max(0.4*imp2-0.05, 0.05), 0.4*imp2)
            a_pns_3 = random.uniform(max(0.4*imp3-0.05, 0.05), 0.4*imp3)
            # Fn
            a_fn_1 = random.uniform(max(0.57*(math.sqrt(imp1))-0.05, 0.05), 0.57*math.sqrt(imp1))
            a_fn_2 = random.uniform(max(0.57*(math.sqrt(imp2))-0.05, 0.05), 0.57*math.sqrt(imp2))
            a_fn_3 = random.uniform(max(0.57*(math.sqrt(imp3))-0.05, 0.05), 0.57*math.sqrt(imp3))
            # Dn
            a_dn_1 = random.uniform(max(0.57*(math.sqrt(imp1))-0.05, 0.05), 0.57*math.sqrt(imp1))
            a_dn_2 = random.uniform(max(0.57*(math.sqrt(imp2))-0.05, 0.05), 0.57*math.sqrt(imp2))
            a_dn_3 = random.uniform(max(0.57*(math.sqrt(imp3))-0.05, 0.05), 0.57*math.sqrt(imp3))

        _users[0].util_fun[server.num][0] = h(imp1,a_pns_1)   # Pns
        _users[0].util_fun[server.num][1] = h(imp1,a_fn_1)   # Fn
        _users[0].util_fun[server.num][2] = h(imp1,a_dn_1)   # Dn
        _users[1].util_fun[server.num][0] = h(imp2,a_pns_2)   # Pns
        _users[1].util_fun[server.num][1] = h(imp2,a_fn_2)   # Fn
        _users[1].util_fun[server.num][2] = h(imp2,a_dn_2)   # Dn
        _users[2].util_fun[server.num][0] = h(imp3,a_pns_3)   # Pns
        _users[2].util_fun[server.num][1] = h(imp3,a_fn_3)   # Fn
        _users[2].util_fun[server.num][2] = h(imp3,a_dn_3)   # Dn
        

# =================================================================== #

urban_servers = copy.deepcopy(servers)
subruban_servers = copy.deepcopy(servers)
rural_servers = copy.deepcopy(servers)

all_servers = {'Urban': urban_servers, 'Suburban': subruban_servers, 'Rural': rural_servers}

# ============================== Game Theory Matching ============================== #

# ============================== Approximate Matching ============================== #

gt_all_users = copy.deepcopy({'Urban': all_users['Urban']})
gt_all_servers = copy.deepcopy({'Urban': all_servers['Urban']})

for area, gt_users in gt_all_users.items(): 
    for user in gt_users:
        # For each user in the GT version, initialize its utility values
        user.current_ptrans = Ptrans_max
        user.current_fn = fn_max
        user.used_datasize = user.datasize
        

gt_times = []
gt_iterations = [None]

for area, gt_users in gt_all_users.items(): 

    gt_start = time.time()

    gt_servers = gt_all_servers[area]

    # Initializing the available servers for each user
    for i in range(N):
        u = gt_users[i]
        u.set_available_servers(gt_servers)
        
    approximate_fedlearner_matching(gt_users, gt_servers)

    print("Approximate FedLearner Matching:\n")

    for u in gt_users:
        allegiance_num = u.get_alligiance().num if u.get_alligiance() is not None else -1
        print("I am User ", u.num, " and I am part of the coalition of Server ", allegiance_num)

    print()

    # ============================== Accurate Matching ============================== #
        
    accurate_fedlearner_matching(gt_users, gt_servers)

    print("Accurate FedLearner Matching:\n")

    for u in gt_users:
        allegiance_num = u.get_alligiance().num if u.get_alligiance() is not None else -1
        print("I am User ", u.num, " and I am part of the coalition of Server ", allegiance_num)

    print()

    gt_end = time.time()

    gt_times.append(gt_end-gt_start)

    print("Game Theory Matching took", gt_end-gt_start, "seconds\n")
    # =============================================================================== #


# ============================== Regret Learning Matching - Complete Information ============================== #

regret_all_users = copy.deepcopy(all_users)
regret_all_servers = copy.deepcopy(all_servers)

regret_times = []
regret_iterations = []

for area, regret_users in regret_all_users.items(): 

    regret_start = time.time()
    
    if area != "Urban":
        if N <= rural_threshold:
            regret_all_users[area] = copy.deepcopy(regret_all_users["Urban"])
            regret_all_servers[area] = copy.deepcopy(regret_all_servers["Urban"])
            regret_times.append(regret_times[0])
            regret_iterations.append(regret_iterations[0])
            continue
        elif N <= suburban_threshold and area == "Suburban":
            regret_all_users[area] = copy.deepcopy(regret_all_users["Urban"])
            regret_all_servers[area] = copy.deepcopy(regret_all_servers["Urban"])
            regret_times.append(regret_times[0])
            regret_iterations.append(regret_iterations[0])
            continue 

    regret_servers = regret_all_servers[area]

    iter = regret_matching(regret_users, regret_servers)

    print("REMORSE Matching:\n")

    for u in regret_users:
        allegiance_num = u.get_alligiance().num if u.get_alligiance() is not None else -1
        print("User:", u.num, "Server:", allegiance_num, "Ptrans:", u.current_ptrans/Ptrans_max, "Fn:", u.current_fn/fn_max, "Dn:", u.used_datasize/(u.datasize))

    print()

    regret_end = time.time()

    regret_times.append(regret_end-regret_start)
    regret_iterations.append(iter)

    print("REMORSE Matching took", regret_end-regret_start, "seconds\n")

# =============================================================================== #

# ============================== Regret Learning Matching - Incomplete Information ============================== #

regretII_all_users = copy.deepcopy(all_users)
regretII_all_servers = copy.deepcopy(all_servers)

regretII_times = []
regretII_iterations = []

for area, regretII_users in regretII_all_users.items(): 

    regret_start = time.time()

    if area != "Urban":
        if N <= rural_threshold:
            regretII_all_users[area] = copy.deepcopy(regretII_all_users["Urban"])
            regretII_all_servers[area] = copy.deepcopy(regretII_all_servers["Urban"])
            regretII_times.append(regretII_times[0])
            regretII_iterations.append(regretII_iterations[0])
            continue
        elif N <= suburban_threshold and area == "Suburban":
            regretII_all_users[area] = copy.deepcopy(regretII_all_users["Urban"])
            regretII_all_servers[area] = copy.deepcopy(regretII_all_servers["Urban"])
            regretII_times.append(regretII_times[0])
            regretII_iterations.append(regretII_iterations[0])
            continue 

    regretII_servers = regretII_all_servers[area]

    iter = regret_matching_II(regretII_users, regretII_servers)

    print("REMORSE Matching II:\n")

    for u in regretII_users:
        allegiance_num = u.get_alligiance().num if u.get_alligiance() is not None else -1
        print("User:", u.num, "Server:", allegiance_num, "Ptrans:", u.current_ptrans/Ptrans_max, "Fn:", u.current_fn/fn_max, "Dn:", u.used_datasize/(u.datasize))

    print()

    regret_end = time.time()

    regretII_times.append(regret_end-regret_start)
    regretII_iterations.append(iter)

    print("REMORSE Matching II took", regret_end-regret_start, "seconds\n")
    
# =============================================================================== #

# Prepare for the Trainings
user_lists = []
server_lists = []

for area, gt_users in gt_all_users.items():
    gt_servers = gt_all_servers[area]
    user_lists.append(gt_users)
    server_lists.append(gt_servers)

for area, regret_users in regret_all_users.items():
    regret_servers = regret_all_servers[area]
    user_lists.append(regret_users)
    server_lists.append(regret_servers)

for area, regret_users in regretII_all_users.items():
    regret_servers = regretII_all_servers[area]
    user_lists.append(regret_users)
    server_lists.append(regret_servers)

# ============================== Federated Learning ============================== #

matching_losses = []
matching_accuracies = []
matching_user_losses = []
matching_user_accuracies = []
matching_FL_times = []

if federated_learning:
    prev_matchings = []

    for _users, _servers in zip(user_lists, server_lists):

        same_matching = check_matching_equality(_servers, prev_matchings)

        get_data = Get_data(_users, _servers)

        X_train, y_train, X_server, y_server = get_data.pre_data()

        fl_time = 0

        if same_matching is not None:
            server_losses, server_accuracy = matching_losses[same_matching], matching_accuracies[same_matching]
            user_losses, user_accuracy = matching_user_losses[same_matching], matching_user_accuracies[same_matching]
            fl_time = matching_FL_times[same_matching]
        else:
            X_train_copy = copy.deepcopy(X_train)
            y_train_copy = copy.deepcopy(y_train)
            X_test_copy = copy.deepcopy(X_server)
            y_test_copy = copy.deepcopy(y_server)
            learning_start = time.time()
            server_losses, server_accuracy, user_losses, user_accuracy = Servers_FL(_users, _servers, rounds, lr, epoch, X_train_copy, y_train_copy, X_test_copy, y_test_copy)
            learning_stop = time.time()
            fl_time = learning_stop - learning_start

        matching_losses.append(server_losses)
        matching_accuracies.append(server_accuracy)
        matching_user_losses.append(user_losses)
        matching_user_accuracies.append(user_accuracy)
        matching_FL_times.append(fl_time)

        for i in range(S):
            print("Server ", i, " achieved:\n")
            print("Loss: ", server_losses[i][-1])
            print("Accuracy: ", server_accuracy[i][-1])
            print()
        print(f"Learning for all 3 Servers took {fl_time/60:.2f} minutes\n")
        print()

        prev_matchings.append(_servers)

    # ================================================================================ #
        
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"\nExecution took {elapsed_time/60:.2f} minutes\n")


# For each Matching log the metrics (Energy, Datarate, Utilities, Payments, Accuracy, Loss)

matchings = []
labels = ["GT_URBAN", "RCI_URBAN", "RCI_SUBURBAN", "RCI_RURAL", "RII_URBAN", "RII_SUBURBAN", "RII_RURAL"]
times = gt_times + regret_times + regretII_times
iterations = gt_iterations + regret_iterations + regretII_iterations

if federated_learning:
    # With Federated Learning
    for i in range(len(labels)):
        matchings.append((user_lists[i], server_lists[i], times[i], iterations[i], matching_losses[i], matching_accuracies[i], matching_user_losses[i], matching_user_accuracies[i], matching_FL_times[i], labels[i]))
else:
    # Without Federated Learning
    for i in range(len(labels)):
        matchings.append((user_lists[i], server_lists[i], times[i], iterations[i], labels[i]))


timestamp = datetime.datetime.now().strftime("%d-%m_%H-%M-%S")

directory_path = "../results"       # Results Directory

# Check if the directory exists
if not os.path.exists(directory_path):
    # If it doesn't exist, create the directory
    os.makedirs(directory_path)

for matching in matchings:

    if federated_learning:
        _users, _servers, duration, iterations, _losses, _accuracies, user_losses, user_accuracies, FL_duration, matching_label = matching     # With Federated Learning
    else:
        _users, _servers, duration, iterations, matching_label = matching     # Without Federated Learning

    # Energy (J)
    mean_Energy = 0
    matched_users = 0
    # Local Energy
    mean_Elocal = 0
    # Datasize
    used_Datasize = 0
    total_Datasize = 0
    # Transfer Energy
    mean_Etransfer = 0
    # Datarate
    mean_Datarate = 0
    # User Utility
    mean_User_Utility = 0

    for user in _users:
        if user.get_alligiance() is not None:
            E_local, payment_fn, payment_dn, datarate, E_transmit = user.get_magnitudes(user.get_alligiance())
            
            # Max Values
            E_local_max, datarate_max, E_transmit_max = user.get_max_values(user.get_alligiance())
            
            # Matched Users
            matched_users += 1
            # Energy
            mean_Energy += E_local * E_local_max
            mean_Energy += E_transmit * E_transmit_max
            # Datasize
            if matching != "GT_URBAN":
                used_Datasize += user.used_datasize
                total_Datasize += user.datasize
            else:
                used_Datasize += user.datasize
                total_Datasize += user.datasize
            # Local Energy
            mean_Elocal += E_local * E_local_max
            # Energy to Transmit
            mean_Etransfer += E_transmit * E_transmit_max
            # Datarate
            mean_Datarate += datarate * datarate_max
            # User Utility
            mean_User_Utility += user_utility_ext(user, user.get_alligiance())
            
    if matched_users != 0:
        mean_Energy /= matched_users
        mean_Elocal /= matched_users
        mean_Etransfer /= matched_users
        mean_Datarate /= matched_users
        mean_User_Utility /= matched_users

    if matching[-1] == "RCI_URBAN":
        user_utility_RCI = mean_User_Utility
    elif matching[-1] == "RII_URBAN":
        user_utility_RII = mean_User_Utility

    # Server Utility
    mean_Server_Utility = 0
    for s in _servers:
        mean_Server_Utility += server_utility_externality(_servers, s.get_coalition(), s)

    mean_Server_Utility /= S

            
    output_filename = f"../results/Areas-u{N}_cp{K}_{timestamp}_{num}.txt"  # Choose a desired filename

    with open(output_filename, 'a') as file:
        file.write(f"Matching: {matching_label}, Users: {N}, Critical Points: {K}\n\
    Mean Energy: {mean_Energy} J\n\
    Mean Elocal: {mean_Elocal} J\n\
    Mean Etransfer: {mean_Etransfer} J\n\
    Mean Datarate: {mean_Datarate} bps\n\
    Mean User Utility: {mean_User_Utility}\n\
    Mean Server Utility: {mean_Server_Utility}\n\
    Percentage of Data Used {used_Datasize/total_Datasize}\n\
    Time: {duration}\n\
    Iterations: {iterations}\n\
    \n")
        
    if federated_learning: 
        with open(output_filename, 'a') as file:
            file.write(f"Fire Server:\n\
        Losses: {_losses[0]}\n\
        Accuracies: {_accuracies[0]}\n")
            for u in _servers[0].get_coalition():
                file.write(f"User {u.num}:\n\
            Losses: {user_losses[u.num]}\n\
            Accuracies: {user_accuracies[u.num]}\n")
            
            file.write(f"Flood Server:\n\
        Losses: {_losses[1]}\n\
        Accuracies: {_accuracies[1]}\n")
            for u in _servers[1].get_coalition():
                file.write(f"User {u.num}:\n\
            Losses: {user_losses[u.num]}\n\
            Accuracies: {user_accuracies[u.num]}\n")
            
            file.write(f"Earthquake Server:\n\
        Losses: {_losses[2]}\n\
        Accuracies: {_accuracies[2]}\n")
            for u in _servers[2].get_coalition():
                file.write(f"User {u.num}:\n\
            Losses: {user_losses[u.num]}\n\
            Accuracies: {user_accuracies[u.num]}\n")
                
            file.write(f"FL Duration: {FL_duration/60:.2f}\n\n")
