from Regret_Matching.utility_function import user_utility_ext
from Classes.User import User, quantized_fn, quantized_datasize, quantized_ptrans
from Classes.Server import Server
from typing import List
from Regret_Matching.Action import Action
from config import N
import numpy as np
import itertools
import random
import copy

convergence_limit = 0.5
ran = random.Random(42)

# Learning rate
def l(t):
    return 1/t**2

# White Noise
def white_noise():
    wn = ran.gauss(mu=0.0, sigma=0.05)
    return wn

# Function to create all possible actions of a user
def matching_actions(servers, quantized_fn, quantized_ptrans, quantized_datasize):
    # Compute the Cartesian product of the input lists
    all_combinations = list(itertools.product(servers, quantized_fn, quantized_ptrans, quantized_datasize))
    all_combinations = [Action(a[0], a[1], a[2], a[3]) for a in all_combinations]
    return all_combinations


def regret_matching_II(users: List[User], servers: List[Server], epsilon = 0):
    t = 1
    actions = matching_actions(servers+[None], quantized_fn, quantized_ptrans, quantized_datasize)

    # Probabilities Initialization
    user_probabilities = [1/((len(servers)+1) * len(quantized_fn) * len(quantized_ptrans) * len(quantized_datasize))] * ((len(servers)+1) * len(quantized_fn) * len(quantized_ptrans) * len(quantized_datasize))
    probabilities = []
    for _ in range(N):
        probs = copy.deepcopy(user_probabilities)
        probabilities.append(probs)
    
    # Regrets Initialization
    regret_vector = []
    for _ in range(N):
        regret_vector.append([0]*len(user_probabilities))

    # Utilities Initialization
    utilities_vector = []
    for user in users:
        servers_copy = copy.deepcopy(servers)
        user_utilities = []
        for action in actions:
            execute_action(user, servers_copy, action)
            user_utilities.append(user_utility_ext(user, user.get_alligiance()) + white_noise())
        utilities_vector.append(user_utilities)

    actions_taken = [None] * len(users)
    actions_taken_indices = [None] * len(users)
    while(not convergence(probabilities) and t < 5000):
        # print("t =",t)
        for u in users:     # For each user get its most likely action
            action_index, repeat = select_action(probabilities[u.num])
            if not repeat:
                execute_action(u, servers, actions[action_index])
                actions_taken[u.num] = actions[action_index]
                actions_taken_indices[u.num] = action_index

        # Calculate current utilities
        current_utilities = [0] * len(users)
        for u in users:
            current_utilities[u.num] = user_utility_ext(u, u.get_alligiance()) + white_noise()

        update_utilities_vector(utilities_vector, actions_taken_indices, users, current_utilities, t) # Update utilities vector
        update_regret_vector(regret_vector, utilities_vector, users, len(actions), current_utilities, t)  # Update regret vector
        update_probabilities(probabilities, regret_vector, users, len(actions), t)   # Update probabilities

        # for u in users:
        #     if not all(p == 0.0 for p in probabilities[u.num]):
        #         max_value = max(probabilities[u.num])
        #         max_index = probabilities[u.num].index(max_value)
        #         print("User", u.num, ", best server:", actions[max_index].target, ", Fn:", actions[max_index].fn, ", Dn:", actions[max_index].ds, ", PTrans:", actions[max_index].ptrans,", max prob:", probabilities[u.num][max_index])
        #     else:
        #         print("User", u.num, ", best server:", actions_taken[u.num].target, ", current server:", u.get_alligiance().num if u.get_alligiance() is not None else None)

        t += 1 

    # After convergence execute the best action for each user
    for user in users:
        if not all(p == 0 for p in probabilities[user.num]):
            max_value = max(probabilities[user.num])
            max_index = probabilities[user.num].index(max_value)
            execute_action(user, servers, actions[max_index])
    
    return t

# Convergence Check
def convergence(probabilities: List[List]):
    for user_probabilities in probabilities:
        prob_sum = sum(user_probabilities)
        found_converging_value = False

        if prob_sum == 0:    # If sum is 0, then the user has converged
            continue
        
        for prob in user_probabilities:
            percentage = prob/prob_sum
            if percentage > convergence_limit:    # Else if you find dominant value the user has converged
                found_converging_value = True
                break

        if found_converging_value == False: # If not then we haven't converged
            return False
        
    return True # If for all users we found dominant values we have converged
                

# Select Action based on probabilities
def select_action(probabilities):
    # If all probabilities for the user are zero, means we chose the best action and
    # except if something changes we have no other reason to select something else
    repeat = all(p == 0 for p in probabilities)
    indices = list(range(len(probabilities)))
    selected_index = ran.choices(indices, weights=probabilities, k=1)[0]
    return selected_index, repeat


# Execute selected Action
def execute_action(user: User, servers: List[Server], action: Action):
    # Matching the server_target of the action to the one of the copied servers
    if action.target is not None:
        actual_target = servers[action.target.num]
    else:
        actual_target = None

    # Change the user's server and make the necessary changes to the server it left from or went to
    if actual_target != user.get_alligiance():
        if user.get_alligiance() is not None:
            user.get_alligiance().remove_from_coalition(user.num)

        if actual_target is not None:
            user.change_server(actual_target)
        else:
            user.change_server(None)

        if actual_target is not None:
            actual_target.add_to_coalition(user)

    # Update the user's parameters based on the action taken
    user.set_parameters(action.fn, action.ptrans, action.ds)

# Update Utilities Vector
def update_utilities_vector(utilities_vector: List[List], actions_taken_indices: List, users: List[User], current_utilities: List, t):
    for user in users:
        # For each user get its current utility 
        current_utility = current_utilities[user.num]
        current_action_index = actions_taken_indices[user.num]
        new_utility_term = 0.5 * (current_utility - utilities_vector[user.num][current_action_index])
        utilities_vector[user.num][current_action_index] += new_utility_term


# Update Regret Vector
def update_regret_vector(regret_vector: List[List], utilities_vector: List[List], users: List[User], num_of_actions, current_utilities: List, t):
    for user in users:
        # For each user get its current utility
        current_utility = current_utilities[user.num]
        for action_index in range(num_of_actions):
            new_regret_term = (utilities_vector[user.num][action_index] - current_utility) - l(t) * regret_vector[user.num][action_index]
            regret_vector[user.num][action_index] += new_regret_term

k_m = 5
# Boltzmann-Gibbs
def boltzmann_gibbs(user_regret_vector: List):
    sum_value = 0
    for regret in user_regret_vector:
        sum_value += np.exp(max(0, regret)/k_m)

    bg_vector = []
    for regret in user_regret_vector:
        bg_vector.append(np.exp(max(0, regret)/k_m)/sum_value)

    return bg_vector

# Update Probabilities
def update_probabilities(probabilities: List[List], regret_vector: List[List], users: List[User], num_of_actions, t):
    for user in users:
        # For each user get its boltzmann_gibbs vector
        bg_vector = boltzmann_gibbs(regret_vector[user.num]) 
        for action_index in range(num_of_actions):
            new_probability_term = (bg_vector[action_index] - probabilities[user.num][action_index])
            probabilities[user.num][action_index] += new_probability_term