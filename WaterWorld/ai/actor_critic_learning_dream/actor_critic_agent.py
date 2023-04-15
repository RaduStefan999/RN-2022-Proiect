import os.path

import keras.models
from keras import backend as backend
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
import numpy as np


class Agent:
    def __init__(self, actor_learning_rate: float, critic_learning_rate: float, gamma: float, nr_of_actions: int,
                 layer_1_size: int, layer_2_size: int, input_dims: int, actor_model: Model = None, critic_model: Model = None,
                 policy: Model = None):

        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.gamma = gamma
        self.nr_of_actions = nr_of_actions
        self.layer_1_size = layer_1_size
        self.layer_2_size = layer_2_size
        self.input_dims = input_dims
        self.actor_model, self.critic_model, self.policy = actor_model, critic_model, policy

        self.possible_actions = list(range(nr_of_actions))

    def build(self):
        self.actor_model, self.critic_model, self.policy = self.build_actor_critic_network()

    def load(self, path: str):
        self.actor_model, self.critic_model, self.policy = self.load_actor_critic_network(path)

    @staticmethod
    def load_actor_critic_network(path: str) -> tuple[Model, Model, Model]:
        return keras.models.load_model(os.path.join(path, "actor_model")), \
               keras.models.load_model(os.path.join(path, "critic_model")), \
               keras.models.load_model(os.path.join(path, "policy_model"))

    def build_actor_critic_network(self) -> tuple[Model, Model, Model]:
        input_layer = Input(shape=self.input_dims)

        dense_layer_1 = Dense(self.layer_1_size, activation="relu")(input_layer)
        dense_layer_2 = Dense(self.layer_2_size, activation="relu")(dense_layer_1)

        actor_probabilities = Dense(self.nr_of_actions, activation="softmax")(dense_layer_2)
        critic_values = Dense(1, activation="linear")(dense_layer_2)

        actor_model = Model(input_layer, actor_probabilities)
        actor_model.compile(optimizer=Adam(learning_rate=self.actor_learning_rate), loss='mean_squared_error')

        critic_model = Model(input_layer, critic_values)
        critic_model.compile(optimizer=Adam(learning_rate=self.critic_learning_rate), loss='mean_squared_error')

        policy = Model(input_layer, actor_probabilities)

        return actor_model, critic_model, policy

    def get_best_action(self, state: np.array) -> int:
        state = state[np.newaxis, :]

        predicted_probability = self.policy.predict(state)[0]
        action = np.random.choice(self.possible_actions, p=predicted_probability)

        return action

    def learn(self, state: np.array, action: int, reward: float, next_state: np.array, is_final: bool) -> None:
        state = state[np.newaxis, :]
        next_state = next_state[np.newaxis, :]

        next_critic_value = self.critic_model.predict(next_state)

        target = reward + self.gamma * next_critic_value * (1 - int(is_final))

        actions_on_hot = np.zeros([1, self.nr_of_actions])
        actions_on_hot[np.arange(1), action] = 1.0

        self.actor_model.fit(state, actions_on_hot)
        self.critic_model.fit(state, target)

    def save_agent(self, path: str) -> None:
        self.actor_model.save(os.path.join(path, "actor_model"))
        self.critic_model.save(os.path.join(path, "critic_model"))
        self.policy.save(os.path.join(path, "policy_model"))
