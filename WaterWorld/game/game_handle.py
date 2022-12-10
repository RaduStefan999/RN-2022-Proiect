import numpy as np
import pygame
from implementations.ple.ple.games.waterworld import WaterWorld


class WaterWorldGame:
    def __init__(self, width: int, height: int, num_creeps: int):
        pygame.init()
        self.current_game = WaterWorld(width=width, height=height, num_creeps=num_creeps)
        self.current_game.screen = pygame.display.set_mode(self.current_game.getScreenDims(), 0, 32)
        self.current_game.clock = pygame.time.Clock()
        self.current_game.rng = np.random.RandomState(24)

    def run(self):
        self.current_game.init()
        while True:
            dt = self.current_game.clock.tick_busy_loop(30)
            self.current_game.step(dt)
            pygame.display.update()


if __name__ == '__main__':
    game = WaterWorldGame(1024, 1024, 20)
    game.run()
