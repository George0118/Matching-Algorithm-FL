from Classes.Server import Server
import math
from Data.load_images import fire_input_paths, earthquake_input_paths, flood_input_paths, count_images
from general_parameters import *
import numpy as np

def dataset_sizes(s: Server, area, users, cps, total_images, verbose = False):
    # For each server count the images and select appropriate number of images to distribute
    if(s.num == 0): 
        image_num = count_images(fire_input_paths)
        ratio = image_num/total_images
        ratio = 1-math.sqrt(ratio)
        image_num = int(1.7*ratio*image_num)
        img_per_usr = image_num/N_max
    elif(s.num == 1):
        image_num = count_images(flood_input_paths)
        ratio = image_num/total_images
        ratio = 1-math.sqrt(ratio)
        image_num = int(1.3*ratio*image_num)
        img_per_usr = image_num/N_max
    else:
        image_num = count_images(earthquake_input_paths)
        ratio = image_num/total_images
        ratio = 1-math.sqrt(ratio)
        image_num = int(0.8*ratio*image_num)
        img_per_usr = image_num/N_max

    # For each user calculate the minimum distance from the relevant Critical Points
    user_min_distances = [-1]*N
    ratios = [0]*N
    
    for u in users:
        for cp in cps:
            user_x, user_y, user_z = u.x, u.y, u.z
            cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

            distance = math.sqrt((cp_x - user_x)**2 + (cp_y - user_y)**2 + (cp_z - user_z)**2)

            if(distance < user_min_distances[u.num] or user_min_distances[u.num] == -1):
                user_min_distances[u.num] = distance

    # Calculate the data size ratios based on the user minimum distance from the CPs
    k = 40
    for i in range(N):
        if user_min_distances[i] <= 0.4:
            ratios[i] = 2 ** (-k * user_min_distances[i])
        else:
            ratios[i] = 0

    ratios = [ratio/max(ratios) for ratio in ratios]

    # Get Sizes
    sizes = [int(1.8 * math.sqrt(math.sqrt(ratio)) * img_per_usr) for ratio in ratios]

    if sum(sizes) > image_num:
        temp_total = sum(sizes)
        sizes = [size*image_num/temp_total for size in sizes]

    if verbose:
        print("Sizes:")
        print(sizes)

    return sizes


def create_zero_list_like(L):
    if isinstance(L, list):
        return [create_zero_list_like(sublist) for sublist in L]
    else:
        return 0
    

def channel_gain(distance, num_user, num_server):   # Channel gain calculation
    g = 128.1 + 37.6 * np.log10(distance) + 8 * random_matrix[num_server][num_user]

    g = 10**(-g / 10)

    return g


# Bell Shaped Utility Functions

# Define the function g(x)
def g(x, imp, a):
    return (2*imp+1) - np.exp(-x/a)

# Define the function f(x)
def f(x, a):
    return (1 / (1 + np.exp(-x/a + 3)))

# Define the function h that returns a function for h(x)
def h(imp, a):
    def h_x(p,n):
        return g(p, imp, a) - f(n, a)
    h_x.imp = imp  # Store parameter imp in the function
    h_x.a = a  # Store parameter a in the function
    return h_x