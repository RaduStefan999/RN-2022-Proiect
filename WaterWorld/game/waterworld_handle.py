import pygame.sprite

from WaterWorld.game.implementations.ple.ple.games.waterworld import WaterWorld
from WaterWorld.game.implementations.ple.ple.games.utils import percent_round_int
from WaterWorld.game.implementations.ple.ple.games.utils.vec2d import vec2d


class Sensor(pygame.sprite.Sprite):
    def __init__(self, init_position: vec2d, sensor_beam_len: float, sensor_beam_width: float, sensor_beam_angle: float):
        super().__init__()
        self.image = pygame.Surface((sensor_beam_width, sensor_beam_len))
        self.image.fill((255, 0, 0))
        self.rect = self.image.get_rect()
        self.rect.midbottom = (init_position.x, init_position.y)

        self.image = pygame.transform.rotate(self.image, sensor_beam_angle)
        current_x, current_y = self.rect.center
        self.rect = self.image.get_rect()
        self.rect.center = (current_x, current_y)
        #self.image = pygame.transform.rotate(self.image, sensor_beam_angle)


    def draw(self, screen) -> None:
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

    def draw(self, screen) -> None:
        for sensor in [self.sensors[5]]:
            sensor.draw(screen)



class WaterWorldHandle(WaterWorld):
    def __init__(self, width: int = 48, height: int = 48, num_creeps: int = 3, enable_sensors=False,
                 sensor_beam_len_percentage: float = 0.17, sensor_beam_width_percentage: float = 0.005, nr_of_sensors: int = 12):
        super().__init__(width, height, num_creeps)

        self.sensor_array = SensorArray(percent_round_int(width, sensor_beam_len_percentage),
                                        percent_round_int(width, sensor_beam_width_percentage),
                                        nr_of_sensors) if enable_sensors else None

    def external_init(self, init_position: tuple) -> None:
        self.sensor_array.build(vec2d(init_position))

    def external_step(self, dt) -> None:
        if self.sensor_array:
            self.sensor_array.draw(self.screen)


