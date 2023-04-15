from keras import models
from keras import layers
from keras import optimizers
import keras
import os.path
import numpy as np
from numpy import random
from WaterWorld.ai.actor_critic_learning_dream.actor_critic_agent import Agent
from WaterWorld.game.game_handle import WaterWorldGame, WaterWorldGameState
from WaterWorld.ai.utils.replay_memory import MemoryManager

def flatten_game_state(game_state: WaterWorldGameState) -> np.array:
    return np.array([*game_state.sensor_output, *game_state.velocity])

flatten_vector = np.vectorize(flatten_game_state)

def get_reward_from_states(lh_game_state: WaterWorldGameState, rh_game_state: WaterWorldGameState) -> float:
    reward = -15
    count_negativ_sensors = 0
    for sensor_output in rh_game_state.sensor_output:
        if sensor_output > 0:
            reward += sensor_output * 10
        else:
            reward += sensor_output * 5
            if sensor_output < 0:
                count_negativ_sensors += 1
    delta_score = rh_game_state.score - lh_game_state.score

    if delta_score > 0:
        reward += delta_score * 100
    else:
        reward += delta_score * 200

    if count_negativ_sensors == 0 and (rh_game_state.position[0] < 60 or rh_game_state.position[0] > 240 or rh_game_state.position[1] < 60 or rh_game_state.position[1] > 240):
        reward += - 100
        print("Sunt bombalau")

    return reward


get_num_action = {
    "left" : 0,
    "up" : 1,
    "right": 2,
    "down" : 3
}

class QNeuronalNetwork:
    # epsilon works like in simulated Annealing
    def __init__(self, number_sensors, epsilon):
        # let's make the neuronal network
        self.nr_sensors = number_sensors
        self.discount = 0.75
        self.memory_handler = MemoryManager(1_000,100_000,200)
        self.epsilon = epsilon
        self.lower_bound_epsilon = 0.001
        self.input_layer = number_sensors + 2
        self.size_first_layer = 512
        self.size_second_layer = 256
        self.size_output_layer = 4
        self.actions_decoder = ["left", "up", "right", "down"]

        self.network = models.Sequential()
        self.network.add(layers.Dense(self.size_first_layer, activation="relu", input_shape=(self.input_layer,)))
        self.network.add(layers.Dense(self.size_second_layer, activation="relu"))
        self.network.add(layers.Dense(self.size_output_layer, activation="linear"))  # left,up,right,down
        optimizer_network = optimizers.RMSprop(learning_rate=0.005)
        self.network.compile(optimizer=optimizer_network, loss='mse')

    # give a state and get the action that the network thinks is the best
    def get_best_action(self, given_state):
        input_neuronal_network = given_state
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



    # train on the bucket u get, here we make the Q-Learning
    def train_bucket(self):
        input_all_sars = self.memory_handler.get_bucket()
        new_states_input = []
        states_input = []
        rewards = []
        actions = []
        for sars_tuple in input_all_sars:
            input_state,action,reward,new_state = sars_tuple
            states_input.append(flatten_game_state(input_state))
            actions.append(action)
            rewards.append(reward)
            new_states_input.append(flatten_game_state(new_state)) 
        rewards = np.array(rewards)
        states_input = (np.array(states_input))
        new_states_input = (np.array(new_states_input))

        #let's get our Q(s',a') so we can calculate the targets
        assert len(new_states_input) == self.memory_handler.bucket_size
        max_Q_values_new_state = self.network.predict(new_states_input)
        max_Q_values_new_state =  np.amax(max_Q_values_new_state, axis=1)
        
        target_values = self.discount * max_Q_values_new_state + rewards
        
        first_stat_Q_values = self.network.predict(states_input)

        print("We have a target values as  : ",target_values[0])

        target_values = self.format_target_vector(actions,first_stat_Q_values,target_values)

        self.network.fit(states_input,target_values,epochs = 1,batch_size = self.memory_handler.bucket_size)


    # here we make our target value for our nn to work on
    def format_target_vector(self,actions,initial_target,values_that_we_target):
        for index in range(self.memory_handler.bucket_size):
            index_action = get_num_action[actions[index]]
            initial_target[index,index_action] = values_that_we_target[index]
        return initial_target

    # we handle the learning from a frame
    def training_frame(self) -> None:
        game_state = self.game_handle.get_game_state()
        input_layer = flatten_game_state(game_state)

        best_action = self.get_best_action(input_layer)
        next_game_state = self.game_handle.act(best_action)

        reward = get_reward_from_states(game_state,next_game_state)
        
        self.memory_handler.add((game_state,best_action,reward,next_game_state))

        if self.memory_handler.enough_elements_to_learn():
            self.train_bucket() # learn from the bucket , in the beginning we are solely getting informations


    #here we make our episode and actually start learning after some time
    def train(self,path)->None:
        self.game_handle = WaterWorldGame(256, 256, 20, self.nr_sensors)
        self.game_handle.init_ai_player()

        nr_of_episodes = 200
        nr_of_iterations_per_game = 300
        
        for it in range(nr_of_episodes):

            current_nr_of_iterations = 0

            while not self.game_handle.game_is_over() and current_nr_of_iterations < nr_of_iterations_per_game:
                current_nr_of_iterations += 1
                print("Episode : ",it," : ")
                self.training_frame()

            self.game_handle.reset_game()
        self.save_agent(path)

    def play(self,path)->None:
        self.game_handle = WaterWorldGame(256, 256, 20, self.nr_sensors)
        self.game_handle.init_ai_player()
        self.game_handle.reset_game()
        self.load(path)

        while not self.game_handle.game_is_over():
            game_state = self.game_handle.get_game_state()

            flattened_game_state = flatten_game_state(game_state)

            chosen_action = self.get_best_action(flattened_game_state)

            self.game_handle.act(chosen_action)

    def save_agent(self,path):
        self.network.save(os.path.join(path,"q_learning_network"))
    
    def load(self,path:str):
        self.network = keras.models.load_model(os.path.join(path,"q_learning_network"))
