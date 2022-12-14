import numpy as np
import pygame
from WaterWorld.game.waterworld_handle import WaterWorldHandle, WaterWorldGameState


class WaterWorldGame:
    def __init__(self, width: int, height: int, num_creeps: int, nr_of_sensors: int, handle: WaterWorldHandle = None):
        pygame.init()
        self.width = width
        self.height = height
        self.num_creeps = num_creeps
        self.nr_of_sensors = nr_of_sensors

        self.current_game = handle

    def init_human_player(self):
        self.current_game = WaterWorldHandle(width=self.width, height=self.height, num_creeps=self.num_creeps, ai_player=False,
                                             enable_sensors=True, nr_of_sensors=self.nr_of_sensors)

        self.current_game.screen = pygame.display.set_mode(self.current_game.getScreenDims(), 0, 32)
        self.current_game.clock = pygame.time.Clock()
        self.current_game.rng = np.random.RandomState(24)

    def init_ai_player(self):
        self.current_game = WaterWorldHandle(width=self.width, height=self.height, num_creeps=self.num_creeps, ai_player=True,
                                             enable_sensors=True, nr_of_sensors=self.nr_of_sensors)

        self.current_game.screen = pygame.display.set_mode(self.current_game.getScreenDims(), 0, 32)
        self.current_game.clock = pygame.time.Clock()
        self.current_game.rng = np.random.RandomState(24)

    def game_is_over(self) -> bool:
        return self.current_game.game_over()

    def reset_game(self) -> None:
        self.current_game.reset()

    def get_actions(self) -> tuple:
        return tuple(self.current_game.actions.keys())

    def get_game_state(self) -> WaterWorldGameState:
        return self.current_game.get_useful_game_state()

    def act(self, action: str, nr_of_frames: int = 1) -> WaterWorldGameState:
        self.__take_action(action, nr_of_frames)
        return self.get_game_state()

    def __take_action(self, action: str, nr_of_frames: int) -> None:
        if self.current_game.game_over():
            return

        for frame in range(nr_of_frames):
            self.current_game.set_ai_action(action)

    def run_interactive(self):
        self.current_game.init()
        while True:
            dt = self.current_game.clock.tick_busy_loop(30)
            self.current_game.step(dt)
            pygame.display.update()


if __name__ == '__main__':
    game = WaterWorldGame(1024, 1024, 20, 12)
    game.init_human_player()
    game.run_interactive()
