import pygame.sprite

from WaterWorld.game.implementations.ple.ple.games.waterworld import WaterWorld
from WaterWorld.game.implementations.ple.ple.games.utils import percent_round_int
from WaterWorld.game.implementations.ple.ple.games.utils.vec2d import vec2d

import math


class WaterWorldGameState:
    def __init__(self, sensor_output: tuple, velocity: tuple, position: tuple, score: int):
        self.sensor_output = sensor_output
        self.velocity = velocity
        self.position = position
        self.score = score


class Sensor(pygame.sprite.Sprite):
    def __init__(self, init_position: vec2d, sensor_beam_len: float, sensor_beam_width: float, sensor_beam_angle: float):
        super().__init__()
        self.image = pygame.Surface((sensor_beam_len * 2, sensor_beam_len * 2))
        self.image.set_colorkey((0, 0, 0))

        self.rect = self.image.get_rect()
        self.rect.center = (init_position.x - sensor_beam_len, init_position.y - sensor_beam_len)
        self.angle = sensor_beam_angle

        self.start_x, self.start_y = sensor_beam_len, sensor_beam_len #init_position.x, init_position.y
        self.sensor_beam_len, self.sensor_beam_width = sensor_beam_len, sensor_beam_width

        self.mask = pygame.mask.from_surface(self.image)

        self.sensor_magnitude = 0

    def draw(self, screen, current_position: vec2d) -> None:
        self.rect.center = (current_position.x - self.sensor_beam_len, current_position.y - self.sensor_beam_len)

        target_x = self.start_x + math.cos(math.radians(self.angle)) * self.sensor_beam_len
        target_y = self.start_y + math.sin(math.radians(self.angle)) * self.sensor_beam_len

        self.image.fill(0)

        color = (255, 255, 0)

        if self.sensor_magnitude > 0:
            color = (0, max(50, 255 * abs(self.sensor_magnitude)), 0)
        elif self.sensor_magnitude < 0:
            color = (max(50, 255 * abs(self.sensor_magnitude)), 0, 0)

        pygame.draw.line(self.image, color, (self.start_x, self.start_y), (target_x, target_y), int(self.sensor_beam_width))
        self.mask = pygame.mask.from_surface(self.image)

        screen.blit(self.image, self.rect.center)


class SensorArray:
    def __init__(self, sensor_beam_len: float, sensor_beam_width: float, nr_of_sensors: int):
        self.sensor_beam_len = sensor_beam_len
        self.sensor_beam_width = sensor_beam_width
        self.nr_of_sensors = nr_of_sensors
        self.sensors = []

    def build(self, init_position: vec2d) -> None:
        for position in range(self.nr_of_sensors):
            self.sensors.append(Sensor(init_position, self.sensor_beam_len, self.sensor_beam_width,
                                       (360 / self.nr_of_sensors) * position))

    def compute_sensor_output(self, screen, creeps, current_position: vec2d, agent_radius: float) -> None:
        for sensor in self.sensors:

            sensor.draw(screen, current_position)

            creep_distances = []
            for creep in creeps:
                offset_x = creep.rect.x - sensor.rect.center[0]
                offset_y = creep.rect.y - sensor.rect.center[1]

                overlap = sensor.mask.overlap(creep.mask, (offset_x, offset_y))

                if not overlap:
                    continue

                distance = math.sqrt((current_position.x - creep.pos.x) ** 2 + (current_position.y - creep.pos.y) ** 2)
                creep_distances.append((creep, distance))

            if not creep_distances:
                sensor.sensor_magnitude = 0
                continue

            min_creep, min_distance = min(creep_distances, key=lambda x: x[1])
            normalized_distance = (max(0.0, min_distance - 2 * agent_radius)) / max(1.0, self.sensor_beam_len - 2 * agent_radius)

            magnitude = math.exp(-normalized_distance)

            sensor.sensor_magnitude = magnitude if min_creep.TYPE == "GOOD" else -magnitude

    def draw(self, screen, current_position: vec2d) -> None:
        for sensor in self.sensors:
            sensor.draw(screen, current_position)

    def get_sensor_output(self) -> tuple:
        sensor_magnitude_list = []
        for sensor in self.sensors:
            sensor_magnitude_list.append(sensor.sensor_magnitude)

        return tuple(sensor_magnitude_list)

    def reset(self) -> None:
        self.sensors = []


class WaterWorldHandle(WaterWorld):
    def __init__(self, width: int = 48, height: int = 48, num_creeps: int = 3, ai_player=False, enable_sensors=False,
                 sensor_beam_len_percentage: float = 0.32, sensor_beam_width_percentage: float = 0.005, nr_of_sensors: int = 12):
        super().__init__(width, height, num_creeps, ai_player)

        self.sensor_array = SensorArray(percent_round_int(width, sensor_beam_len_percentage),
                                        percent_round_int(width, sensor_beam_width_percentage),
                                        nr_of_sensors) if enable_sensors else None

    def external_init(self, init_position: tuple) -> None:
        if self.sensor_array:
            self.sensor_array.build(vec2d(init_position))

    def external_step(self, dt) -> None:
        if self.sensor_array:
            self.sensor_array.compute_sensor_output(self.screen, self.creeps, self.player.pos, self.AGENT_RADIUS)
            self.sensor_array.draw(self.screen, self.player.pos)

    def reset(self) -> None:
        if self.sensor_array:
            self.sensor_array.reset()

        self.init()

    def get_useful_game_state(self) -> WaterWorldGameState:
        return WaterWorldGameState(self.sensor_array.get_sensor_output(), (self.player.vel.x, self.player.vel.y),
                                   (self.player.pos.x, self.player.pos.y), self.score)


