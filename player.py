import pygame
import os
from config import agent_size_x, agent_size_y


class Player(pygame.sprite.Sprite):
    def __init__(self, starting_pos, scaling, tiny_vis=False):
        super().__init__()

        self.animations = {
            "idle": pygame.image.load(
                os.path.join("assets/spaceship/idle", "spaceship_master.png")
            ).convert_alpha(),
            "left": pygame.image.load(
                os.path.join("assets/spaceship/left", "spaceship_master_left_turn.png")
            ).convert_alpha(),
            "right": pygame.image.load(
                os.path.join("assets/spaceship/right", "spaceship_master_right_turn.png")
            ).convert_alpha(),
        }
        self.animations["idle"] = pygame.transform.scale(
            self.animations["idle"], (agent_size_x * scaling, agent_size_y * scaling)
        )
        self.animations["left"] = pygame.transform.scale(
            self.animations["left"], (agent_size_x * scaling, agent_size_y * scaling)
        )
        self.animations["right"] = pygame.transform.scale(
            self.animations["right"], (agent_size_x * scaling, agent_size_y * scaling)
        )
        self.image = self.animations["idle"]
        if tiny_vis:
            self.image = pygame.Surface(
                (agent_size_x * scaling, agent_size_y * scaling)
            )
            self.image.fill("green")
        self.rect = self.image.get_rect(topleft=starting_pos)

        self.direction = pygame.math.Vector2(0, 0)
        self.drift = pygame.math.Vector2(0, 0)
        self.crashed = False

    def get_input(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:
            self.direction.x = 1
        elif keys[pygame.K_LEFT]:
            self.direction.x = -1
        else:
            self.direction.x = 0

    def update(self, player_position, scaling, keyboard_input=False):
        if keyboard_input:
            self.get_input()
            player_horizontal_movement = self.direction.x + self.drift.x
            self.rect.x += player_horizontal_movement * scaling
        else:
            self.rect = self.image.get_rect(topleft=player_position)

    def animate(self, direction):
        if direction is None:
            self.image = self.animations["idle"]
        elif direction == "Left":
            self.image = self.animations["left"]
        elif direction == "Right":
            self.image = self.animations["right"]

    def approach(self, speed, scaling):
        self.rect.y += speed * scaling * 2 / 3