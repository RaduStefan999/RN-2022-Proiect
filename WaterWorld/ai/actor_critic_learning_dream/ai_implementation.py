import math

from WaterWorld.ai.actor_critic_learning_dream.actor_critic_agent import Agent
from WaterWorld.game.game_handle import WaterWorldGame, WaterWorldGameState
from WaterWorld.ai.utils.replay_memory import MemoryManager
import numpy as np


def flatten_game_state(game_state: WaterWorldGameState) -> np.array:
    return np.array([*game_state.sensor_output, *game_state.velocity])


def get_reward_from_states(lh_game_state: WaterWorldGameState, rh_game_state: WaterWorldGameState) -> float:
    reward = -2
    count_negativ_sensors = 0
    for sensor_output in rh_game_state.sensor_output:
        if sensor_output > 0:
            reward += sensor_output * 4
        else:
            reward += sensor_output * 2
            if sensor_output < 0:
                count_negativ_sensors += 1
    delta_score = rh_game_state.score - lh_game_state.score

    if delta_score > 0:
        reward += delta_score * 600
    else:
        reward += delta_score * 1200

    if count_negativ_sensors == 0 and (rh_game_state.position[0] < 60 or rh_game_state.position[0] > 240 or rh_game_state.position[1] < 60 or rh_game_state.position[1] > 240):
        reward += - 100
        print("Sunt bombalau")

    return reward


class AiImplementation:
    def __init__(self, nr_of_sensors: int, game_handle: WaterWorldGame = None):
        self.nr_of_sensors = nr_of_sensors
        self.agent = Agent(0.0001, 0.0005, 0.99, 4, 1024, 512, nr_of_sensors + 2)
        self.game_handle = game_handle

        self.actions = {
            0: "up",
            1: "left",
            2: "right",
            3: "down"
        }

    def train(self) -> None:
        self.agent.build()

        self.game_handle = WaterWorldGame(256, 256, 10, self.nr_of_sensors)
        self.game_handle.init_ai_player()
        self.game_handle.reset_game()

        memory_manager = MemoryManager(1000, 10000, 1)

        nr_of_episodes = 6
        nr_of_iterations_per_game = 300

        for it in range(nr_of_episodes):
            score = 0

            current_nr_of_iterations = 0

            while not self.game_handle.game_is_over() and current_nr_of_iterations < nr_of_iterations_per_game:
                current_nr_of_iterations += 1

                if current_nr_of_iterations:
                    self.training_step(memory_manager)

            self.game_handle.reset_game()

    def training_step(self, memory_manager: MemoryManager) -> None:
        game_state = self.game_handle.get_game_state()
        flattened_game_state = flatten_game_state(game_state)

        chosen_action = self.agent.get_best_action(flattened_game_state)
        translated_action = self.actions[chosen_action]

        next_game_state = self.game_handle.act(translated_action)

        reward = get_reward_from_states(game_state, next_game_state)

        memory_manager.add((game_state, chosen_action, reward, next_game_state))

        if memory_manager.enough_elements_to_learn():
            learn_game_state, learn_chosen_action, learn_reward, learn_next_game_state = \
                memory_manager.get_random_element_history()

            self.agent.learn(flatten_game_state(learn_game_state), learn_chosen_action, reward,
                             flatten_game_state(next_game_state), False)

    def save_model(self, path: str) -> None:
        self.agent.save_agent(path)

    def load(self, path: str) -> None:
        self.game_handle = WaterWorldGame(256, 256, 10, self.nr_of_sensors)
        self.game_handle.init_ai_player()
        self.game_handle.reset_game()

        self.agent.load(path)

    def play(self):
        while not self.game_handle.game_is_over():
            game_state = self.game_handle.get_game_state()

            flattened_game_state = flatten_game_state(game_state)

            chosen_action = self.agent.get_best_action(flattened_game_state)
            translated_action = self.actions[chosen_action]

            self.game_handle.act(translated_action)


def train():
    ai_implementation = AiImplementation(24)
    ai_implementation.train()
    ai_implementation.save_model(r"D:\personal\Facultate\RN\Proiect\RN-2022-Proiect\SavedModels")


def play():
    ai_implementation = AiImplementation(24)
    ai_implementation.load(r"D:\personal\Facultate\RN\Proiect\RN-2022-Proiect\SavedModels")
    ai_implementation.play()


if __name__ == '__main__':
    #train()
    play()





