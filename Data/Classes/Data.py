from config import S
from Data.load_images import load_images, load_neutral_images
from Data.load_images import earthquake_input_paths, fire_input_paths, flood_input_paths
from general_parameters import N_max
from sklearn.utils import shuffle
import numpy as np

# to get the training data, and split the data via the number of clients
class Get_data:
  def __init__(self, users, servers):
    self.users = users    # clients
    self.servers = servers    # servers
    self.n = len(users)   # num of clients

  def load_data(self):

    # Load and shuffle fire data
    print("Loading Images for Fires...")
    X_train_fire, X_server_fire, y_train_fire, y_server_fire, fire_img_num = load_images(fire_input_paths, "fire")
    print("Fire Images loaded!\n")

    # Load and shuffle flood data
    print("Loading Images for Floods...")
    X_train_flood, X_server_flood, y_train_flood, y_server_flood, flood_img_num = load_images(flood_input_paths, "flood")
    print("Flood Images loaded!\n")

    # Load and shuffle earthquake data
    print("Loading Images for Earthquakes...")
    X_train_earthquake, X_server_earthquake, y_train_earthquake, y_server_earthquake, earthquake_img_num = load_images(earthquake_input_paths, "earthquake")
    print("Earthquake Images loaded!\n")

    return (X_train_fire, X_server_fire, y_train_fire, y_server_fire, fire_img_num) ,\
           (X_train_flood, X_server_flood, y_train_flood, y_server_flood, flood_img_num),\
           (X_train_earthquake, X_server_earthquake, y_train_earthquake, y_server_earthquake, earthquake_img_num)


  def split_data(self, data, server, img_num): 
    users = self.users
    critical_points = server.get_critical_points()
    user_min_distances = [-1]*len(users)

    for u in users:
      for cp in critical_points:
        user_x, user_y, user_z = u.x, u.y, u.z
        cp_x, cp_y, cp_z = cp.x, cp.y, cp.z

        distance = np.sqrt((cp_x - user_x)**2 + (cp_y - user_y)**2 + (cp_z - user_z)**2)

        if(distance < user_min_distances[u.num] or user_min_distances[u.num] == -1):
          user_min_distances[u.num] = distance

    ratios = [None]*len(users)

    # Calculate the data size ratios based on the user minimum distance from the CPs
    for i in range(len(users)):
        if user_min_distances[i] <= 0.4:
            ratios[i] = 1/(user_min_distances[i] + 1e-6)
        else:
            ratios[i] = 0

    ratios = [ratio/max(ratios) for ratio in ratios]

    def_len = img_num/N_max

    # Get Sizes
    sizes = [int(1.8 * np.sqrt(np.sqrt(ratio)) * def_len) for ratio in ratios]

    s_data = []
    
    for i, size in enumerate(sizes):
        c_data = data[i][:size]
        s_data.append(c_data)
    return s_data

  def pre_data(self):

    (X_train_fire, X_server_fire, y_train_fire, y_server_fire, fire_img_num) ,\
    (X_train_flood, X_server_flood, y_train_flood, y_server_flood, flood_img_num),\
    (X_train_earthquake, X_server_earthquake, y_train_earthquake, y_server_earthquake, earthquake_img_num) = self.load_data()

    # Split fire data to each user
    X_train_fire=self.split_data(X_train_fire, self.servers[0], fire_img_num) 
    y_train_fire=self.split_data(y_train_fire, self.servers[0], fire_img_num)

    # Split flood data to each user
    X_train_flood=self.split_data(X_train_flood, self.servers[1], flood_img_num) 
    y_train_flood=self.split_data(y_train_flood, self.servers[1], flood_img_num)

    # Split earthquake data to each user
    X_train_earthquake=self.split_data(X_train_earthquake, self.servers[2], earthquake_img_num) 
    y_train_earthquake=self.split_data(y_train_earthquake, self.servers[2], earthquake_img_num)

    print("Splited Successfully\n")

    # Concatenate training separated data
    X_train = [concatenate_non_empty_arrays((X_train_fire[i], X_train_flood[i], X_train_earthquake[i])) for i in range(self.n)]
    y_train = [concatenate_non_empty_arrays((y_train_fire[i], y_train_flood[i], y_train_earthquake[i])) for i in range(self.n)]

    print("Loading Neutral Images...")
    X_train_neutral, y_train_neutral, X_server_neutral, y_server_neutral = load_neutral_images()
    print("Neutral Images loaded!\n")

    X_train = [np.concatenate((X_train[i], X_train_neutral[i])) for i in range(self.n)]
    y_train = [np.concatenate((y_train[i], y_train_neutral[i])) for i in range(self.n)]

    # Construct server data
    X_server = [np.concatenate((X_server_fire, X_server_neutral[0])),
                np.concatenate((X_server_flood, X_server_neutral[1])),
                np.concatenate((X_server_earthquake, X_server_neutral[2]))]
    y_server = [np.concatenate((y_server_fire, y_server_neutral[0])),
                np.concatenate((y_server_flood, y_server_neutral[1])),
                np.concatenate((y_server_earthquake, y_server_neutral[2]))]

    # Shuffle the training Data of each user
    for i in range(self.n):
       X_train[i], y_train[i] = shuffle(X_train[i], y_train[i], random_state=42)

    # Shuffle Data for each Server
    for i in range(S):
       X_server[i], y_server[i] = shuffle(X_server[i], y_server[i], random_state=42)

    print("Data Concatenated and Shuffled Successfully\n")

    # for i in range(self.n):
    #   print("I am user ", i, " and my datasize is: ", len(X_train[i]))
    #   print("My distribution is:")
    #   print("Fires: ", len(X_train_fire[i]))
    #   print("Floods: ", len(X_train_flood[i]))
    #   print("Earthquakes: ", len(X_train_earthquake[i]))
    #   print()

    return X_train, y_train, X_server, y_server
  

def concatenate_non_empty_arrays(arrays):
    non_empty_arrays = [arr for arr in arrays if len(arr) > 0]
    if non_empty_arrays:
        return np.concatenate(non_empty_arrays)
    else:
        return np.array([])
  
  