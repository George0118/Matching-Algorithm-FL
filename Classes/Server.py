import numpy as np
import math
from Data.Classes.Model import Model
from general_parameters import N, current_max_users

# Server Class
class Server:
    def __init__(self, x, y, z, p, num):
        self._x = x
        self._y = y
        self._z = z
        self._p = p
        self._min_payment = 1
        self._Ns_max = p/self._min_payment
        self._num = num
        self.critical_points = []
        self.coalition = set()
        self.invitations = set()

    def set_p(self, value):         # Set payment
        self._p = value

    def get_invitation(self, user):         # Getting Invited by user
        self.invitations.add(user)

    def add_to_coalition(self, user):         # Add user to coalition
        self.coalition.add(user)

    def remove_from_coalition(self, num):     # Remove user from coalition
        user_to_remove = next((user for user in self.coalition if user.num == num), None)
        self.coalition.remove(user_to_remove)

    def add_critical_point(self, cp):
        self.critical_points.append(cp)

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def z(self):
        return self._z
    
    @property
    def p(self):
        return self._p
    
    @property
    def num(self):
        return self._num
    
    @property
    def Ns_max(self):
        return self._Ns_max
    
    def get_coalition(self):
        return self.coalition
    
    def get_invitations_list(self):
        return self.invitations
    
    def clear_invitations_list(self):
        self.invitations = set()

    def get_critical_points(self):
        return self.critical_points


    # ============= Federated Learning ============== #
    
    def scale_model_weights(self,weight,scalar,num):
        '''function for scaling a models weights'''
        weight_final = []
        wei=[]
        steps = 8

        for j in range(num):
            fa=scalar[j]
            scar=[fa for k in range(steps)]
            for i in range(steps):
                weight_final.append(scar[i]*weight[j][i])
                #weight_final.append(weight[i])
            wei.append(weight_final)
        return wei

    def sum_scaled_weights(self,scaled_weight_lists):
        '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
        avg_weights = [sum(weights) for weights in zip(*scaled_weight_lists)]
        return avg_weights

    def evaluate(self,model,test_X, test_y):
        loss,acc=Model.evaluate_model(model,test_X, test_y)
        return loss,acc

    def specify_disaster(self, labels):
        disaster = ''
        if self.num == 0:
            disaster = 'fire'
        elif self.num == 1:
            disaster = 'flood'
        elif self.num == 2:
            disaster = 'earthquake'

        if disaster == '':
            return labels

        labels = np.where(labels == disaster, 1, 0)

        return labels
    
    
    def factors_calculation(self):
        coalition = self.get_coalition()
        critical_points = self.get_critical_points()

        factors = [0] * N
        max_factor = None

        for u in coalition:
            importance_list = u.get_importance()
            factor = 0
            for cp in critical_points:
                factor += (importance_list[cp.num]**0.33) * u.used_datasize
            factors[u.num] = factor
            if max_factor is None or max_factor < factor:
                max_factor = factor

        fact_sum = 0
        for u in coalition:
            factors[u.num] = np.exp((factors[u.num] - max_factor)/max_factor)
            fact_sum += factors[u.num]

        for u in coalition:
            factors[u.num] /= fact_sum
        
        return factors
