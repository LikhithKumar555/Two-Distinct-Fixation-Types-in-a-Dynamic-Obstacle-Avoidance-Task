import pygame
from config import wall_size


class Wall(pygame.sprite.Sprite):
    def __init__(self, pos, scaling):
        super().__init__()
        self.image = pygame.Surface((wall_size * scaling, wall_size * scaling))
        self.image.fill("white")
        self.rect = self.image.get_rect(topleft=pos)

    def update(self, speed, scaling, horizontal_movement):
        # vertical movement
        self.rect.y -= (1 * scaling) * speed
        # horizontal movement
        self.rect.x += horizontal_movement * scaling * speed
