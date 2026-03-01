from Regret_Matching.utility_function import user_utility_ext
from Classes.User import User, quantized_fn, quantized_datasize, quantized_ptrans
from Classes.Server import Server
from typing import List
from Regret_Matching.Action import Action
from config import N
import itertools
import random
import copy

convergence_limit = 0.5
ran = random.Random(42)

# Learning rate
def l(t):
    return 1/t**0.8

# Function to create all possible actions of a user
def matching_actions(servers, quantized_fn, quantized_ptrans, quantized_datasize):
    # Compute the Cartesian product of the input lists
    all_combinations = list(itertools.product(servers, quantized_fn, quantized_ptrans, quantized_datasize))
    all_combinations = [Action(a[0], a[1], a[2], a[3]) for a in all_combinations]
    return all_combinations


def regret_matching(users: List[User], servers: List[Server], epsilon = 0):
    t = 1
    actions = matching_actions(servers+[None], quantized_fn, quantized_ptrans, quantized_datasize)

    # Probabilities Initialization
    user_probabilities = [1/((len(servers)+1) * len(quantized_fn) * len(quantized_ptrans) * len(quantized_datasize))] * ((len(servers)+1) * len(quantized_fn) * len(quantized_ptrans) * len(quantized_datasize))
    probabilities = []
    for _ in range(N):
        probabilities.append(copy.deepcopy(user_probabilities))
    
    # Regret Vector Initialization
    regret_vector = []
    for _ in range(N):
        regret_vector.append([0]*len(user_probabilities))

    actions_taken = [None] * len(users)
    while(not convergence(probabilities) and t < 5000):
        # print("t =",t)
        for u in users:     # For each user get its most likely action
            action_index, repeat = select_action(probabilities[u.num])
            if not repeat:
                execute_action(u, servers, actions[action_index])
                actions_taken[u.num] = actions[action_index]

        update_regret_vector(regret_vector, users, servers, actions, t, actions_taken)  # Update regret vector
        update_probabilities(probabilities, regret_vector, users)   # Update probabilities

        # for u in users:
        #     if not all(p == 0.0 for p in probabilities[u.num]):
        #         max_value = max(probabilities[u.num])
        #         max_index = probabilities[u.num].index(max_value)
        #         print("User", u.num, ", best server:", actions[max_index].target.num, ", Fn:", actions[max_index].fn, ", Dn:", actions[max_index].ds, ", PTrans:", actions[max_index].ptrans,", max prob:", probabilities[u.num][max_index])
        #     else:
        #         print("User", u.num, ", best server:", actions_taken[u.num].target.num, ", current server:", u.get_alligiance().num if u.get_alligiance() is not None else None)

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

# Regret Update Rule
def update_regret(t, user: User, servers: List[Server], current_utility, other_action: Action, external_denominators: List):
    term2 = current_utility
    execute_action(user, servers, other_action) # Execute action
    if other_action.target is None:
        term1 = 0
    else:
        term1 = user_utility_ext(user, user.get_alligiance(), external_denominator=external_denominators[other_action.target.num])   # And get new utility

    result = (term1 - term2)

    return result

import time
# Update Regret Vector
def update_regret_vector(regret_vector, users: List[User], servers: List[Server], actions: List[Action], t, actions_taken: List[Action]):
    # Copy users and servers so that changes to calculate regrets dont affect our original users and servers
    users_copy = copy.deepcopy(users)
    servers_copy = copy.deepcopy(servers)

    # Set correct pointers from copied users to copied servers and back
    # Empty copied coalitions
    for server in servers_copy:
        users_to_remove = []
        for u in server.get_coalition():
            users_to_remove.append(u.num)
        for u_num in users_to_remove:
            server.remove_from_coalition(u_num)

    # Repopulate coalitions with copied users
    for user in users_copy:
        if user.get_alligiance() is not None:
            servers_copy[user.get_alligiance().num].add_to_coalition(user)

    # Set copied servers as users' alligiances
    for user in users_copy:
        if user.get_alligiance() is not None:
            server_num = user.get_alligiance().num
            user.change_server(servers_copy[server_num])

    time1 = time.time()
    for user in users_copy:
        # For each user get its current utility and also calculate the externality of the rest of the users (for datarate, E_transmit)
        current_external_denominators = user.get_external_denominators(servers_copy)
        if user.get_alligiance() is not None:
            current_utility = user_utility_ext(user, user.get_alligiance(),
                                             current_external_denominators[user.get_alligiance().num])
        else:
            current_utility = 0
        for action_index, action in enumerate(actions):
            new_regret_term = update_regret(t, user, servers_copy, current_utility, action, current_external_denominators)
            regret_vector[user.num][action_index] =  (1-l(t)) * regret_vector[user.num][action_index] + new_regret_term

        # Reset User
        execute_action(user, servers_copy, actions_taken[user.num])
        
    time2 = time.time()

    # print("Time:", time2 - time1)

# Update Probabilities
def update_probabilities(probabilities: List[List], regret_vector: List[List], users: List[User]):
    for user in users:
        sum = 0
        for regret in regret_vector[user.num]:
            sum += max(0,regret)

        for index, regret in enumerate(regret_vector[user.num]):
            if sum == 0:
                probabilities[user.num][index] = 0
            else:
                probabilities[user.num][index] = max(0,regret)/sum