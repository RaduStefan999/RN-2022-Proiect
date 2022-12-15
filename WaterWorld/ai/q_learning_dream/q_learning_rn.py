from keras import models
from keras import layers
from keras import optimizers
import numpy as np
from numpy import random


class QNeuronalNetwork:
    # epsilon works like in simulated Annealing
    def __init__(self, number_sensors, epsilon, memory_handler):
        # let's make the neuronal network
        self.memory_handler = memory_handler
        self.epsilon = epsilon
        self.lower_bound_epsilon = 0.1
        self.input_layer = number_sensors + 2
        self.size_first_layer = 20
        self.size_second_layer = 12
        self.size_output_layer = 4
        self.actions_decoder = ["left", "up", "right", "down"]

        self.network = models.Sequential()
        self.network.add(layers.Dense(self.size_first_layer, activation="relu", input_shape=(self.input_layer,)))
        self.network.add(layers.Dense(self.size_second_layer, activation="relu"))
        self.network.add(layers.Dense(self.size_output_layer, activation="linear"))  # left,up,right,down
        optimizer_network = optimizers.RMSprop(learning_rate=0.001)
        self.network.compile(optimizer=optimizer_network, loss='mse')

    # give a state and get the action that the network thinks is the best
    def get_best_action(self, given_state):
        input_neuronal_network = given_state[0] + given_state[1]
        input_neuronal_network = np.array(input_neuronal_network).reshape(1, self.input_layer)

        last_layer_values = self.network.predict(input_neuronal_network)[0]  # bcs keras:D
        index = np.argmax(last_layer_values)
        print(last_layer_values)

        hit_value = random.rand()
        if hit_value <= max(self.lower_bound_epsilon, self.epsilon):  # we have a hit
            self.epsilon = 0.999 * self.epsilon
            decisions_pos = [0, 1, 2, 3].remove(index)
            random_choice = random.randint(2)
            return self.actions_decoder[random_choice]

        return self.actions_decoder[index]

    # train on the bucket u get
    def train(self):
        training_input = self.memory_handler.get_bucket()
        print(training_input)