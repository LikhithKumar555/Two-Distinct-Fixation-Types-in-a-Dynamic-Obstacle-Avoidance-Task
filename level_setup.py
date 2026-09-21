import pygame
import numpy as np
import pandas as pd
import random
from comets import Comet
from player import Player
from walls import Wall
from drift_tiles import DriftTile
from particles import Particle
from lines import Line
from displays import display_soc_question
from config import *
from draw_transparent_shapes import draw_rect_alpha, draw_polygon_alpha

display_keys = False
question_soc = True

class Level:
    def __init__(
        self,
        wall_list,
        obstacles_list,
        player_starting_position,
        drift_ranges,
        screen,
        scaling,
        code,
        FPS=30,
        eye_tracker_device=None,
        gaze_queue=None,
        n_run=0,
        tiny_vis=False,
        keyboard_input=False,
        trial=0,
        attempt=0,
        input_noise_magnitude=None,
        input_noise_threshold=0,
        drift_enabled=False,
        participant_id="unknown",
        gaze_output_file=None,
    ):

        self.code = code
        self.SoC = None
        self.trial = trial
        self.attempt = attempt
        self.display_surface = screen
        self.level_size_y = "undetermined"
        self.setup_level(
            wall_list,
            obstacles_list,
            player_starting_position,
            drift_ranges,
            drift_enabled,
            scaling,
            tiny_vis,
            keyboard_input,
        )
        self.direction = pygame.math.Vector2(0, 0)
        self.drift = pygame.math.Vector2(0, 0)
        self.current_input = None
        self.horizontal_movement = 0
        self.transparency_left = 90
        self.transparency_right = 90
        self.visible_obstacles = []
        self.adjacent_wall_tiles_x_pos = []
        self.drift_enabled = drift_enabled
        self.input_noise_threshold = input_noise_threshold * scaling
        self.input_noise_magnitude = input_noise_magnitude
        self.input_noise_on = False
        self.n_run = n_run
        self.participant_id = participant_id
        self.gaze_output_file = gaze_output_file
        self.frames_collision_threshold = FPS / 10
        self.frames_with_collision = 0
        self.time_played = 0
        self.FPS = FPS
        self.eye_tracker_device = eye_tracker_device
        self.level_done = False
        self.quit = False

        
        self.gaze_queue = gaze_queue
        self.current_gaze = None

        self.columns = [
            "trial", "attempt", "time_played", "time_tag", "level_size_y",
            "player_pos", "collision", "current_input", "drift_enabled",
            "current_drift", "level_done", "input_noise_magnitude",
            "input_noise_on", "visible_obstacles", "adjacent_wall_tiles_x_pos",
            "visible_drift_tiles", "SoC", "gaze_x", "gaze_y"
        ]
        self.data = pd.DataFrame(columns=self.columns)

        if self.drift_enabled:
            self.drift_abbr = 'T'
        else:
            self.drift_abbr = 'F'
        if not input_noise_magnitude:
            self.input_noise_abbr = 'N'
        else:
            self.input_noise_abbr = str(input_noise_magnitude[0]).upper()

    def _get_output_filename(self):
        if self.gaze_output_file:
            return self.gaze_output_file
        else:
            return f"data/{self.code}_output_{self.trial}{self.drift_abbr}{self.input_noise_abbr}_{self.n_run:0>2}.csv"

    def setup_level(self, wall_list, obstacles_list, player_starting_position,
                    drift_ranges, drift_enabled, scaling, tiny_vis, keyboard_input):
        self.walls = pygame.sprite.Group()
        self.comets = pygame.sprite.Group()
        self.drift_tiles = pygame.sprite.Group()
        self.player = pygame.sprite.GroupSingle()
        self.particles = pygame.sprite.Group()
        self.bottom_edge = pygame.sprite.GroupSingle()
        self.finish_line = pygame.sprite.GroupSingle()

        for i in range(1, len(wall_list) + 1):
            left_wall_x_pos = wall_list[str(i)][0]
            left_wall = Wall((left_wall_x_pos, i * scaling), scaling)
            right_wall_x_pos = wall_list[str(i)][1]
            right_wall = Wall((right_wall_x_pos, i * scaling), scaling)
            self.walls.add(left_wall, right_wall)

        last_wall_tile = self.walls.sprites()[-1]
        self.level_size_y = last_wall_tile.rect.y + wall_size * scaling

        for key in obstacles_list:
            comet_sprite = Comet((key["x"], key["y"]), key["size"])
            self.comets.add(comet_sprite)

        if keyboard_input:
            player_appearance = [player_starting_position[0], (player_starting_position[1] - pre_trial_steps * scaling)]
            player_sprite = Player(player_appearance, scaling, tiny_vis)
            self.player.add(player_sprite)
        else:
            player_sprite = Player(player_starting_position, scaling, tiny_vis)
            self.player.add(player_sprite)

        if drift_enabled:
            for i in range(len(drift_ranges)):
                drift_info = drift_ranges[i]
                drift_tile = DriftTile(drift_info[0], drift_info[1], drift_info[2], drift_info[3], scaling)
                self.drift_tiles.add(drift_tile)

        for _ in range(int(last_wall_tile.rect.y / scaling * 2.5)):
            x_pos = np.random.uniform(low=edge * scaling, high=level_size_x * scaling + edge * scaling, size=1)
            y_pos = np.random.uniform(low=0, high=self.level_size_y, size=1)
            particle_tile = Particle((x_pos[0], y_pos[0]), random.choice(particle_sizes), scaling)
            self.particles.add(particle_tile)

        bottom_edge_tile = Line([0, (observation_space_size_y - bottom_edge) * scaling],
                                [(level_size_x + 2 * edge) * scaling, bottom_edge * scaling])
        self.bottom_edge.add(bottom_edge_tile)

        finish_line_tile = Line(pos=[edge * scaling, last_wall_tile.rect.y],
                                size=[level_size_x * scaling, scaling], col="seagreen")
        self.finish_line.add(finish_line_tile)

    def update_gaze(self):
        if self.gaze_queue is not None:
            while not self.gaze_queue.empty():
                self.current_gaze = self.gaze_queue.get_nowait()

    def get_input(self):
        player = self.player.sprite
        input_noise = 0
        if player.rect.y > self.input_noise_threshold:
            self.input_noise_on = True
            mu = 0
            if self.input_noise_magnitude == "weak":
                sigma = 0.5
                input_noise = np.random.normal(mu, sigma, 1)
            elif self.input_noise_magnitude == "medium":
                sigma = 1.0
                input_noise = np.random.normal(mu, sigma, 1)
            elif self.input_noise_magnitude == "strong":
                sigma = 1.5
                input_noise = np.random.normal(mu, sigma, 1)
            elif self.input_noise_magnitude == "very_strong":
                sigma = 2.0
                input_noise = np.random.normal(mu, sigma, 1)
            else:
                sigma = 0
                input_noise = np.random.normal(mu, sigma, 1)
        else:
            self.input_noise_on = False

        self.transparency_left = 90
        self.transparency_right = 90
        
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT] and keys[pygame.K_RIGHT]:
            self.transparency_right = 150
            self.transparency_left = 150
            self.current_input = None
            self.direction.x = 0
        elif keys[pygame.K_RIGHT]:
            self.current_input = "Right"
            self.direction.x = -1 + input_noise
            self.transparency_right = 150
        elif keys[pygame.K_LEFT]:
            self.current_input = "Left"
            self.direction.x = 1 + input_noise
            self.transparency_left = 150
        else:
            self.current_input = None
            self.direction.x = 0

    def update(self):
        self.get_input()
        self.horizontal_movement = self.direction.x + self.drift.x

    def check_for_collision(self):
        player = self.player.sprite
        if (player.rect.collidelist(self.comets.sprites()) > -1 or
            player.rect.collidelist(self.walls.sprites()) > -1):
            self.frames_with_collision += 1
        else:
            self.frames_with_collision = 0
        if self.frames_with_collision > self.frames_collision_threshold:
            player.crashed = True

    def check_for_drift(self):
        player = self.player.sprite
        self.drift.x = 0
        for sprite in self.drift_tiles.sprites():
            if sprite.rect.left > player.rect.right:
                if player.rect.top in range(sprite.rect.top, sprite.rect.bottom):
                    self.drift.x = -1 / sprite.direction
                elif player.rect.bottom in range(sprite.rect.top, sprite.rect.bottom):
                    self.drift.x = -1 / sprite.direction
            elif sprite.rect.right < player.rect.left:
                if player.rect.top in range(sprite.rect.top, sprite.rect.bottom):
                    self.drift.x = -1 / sprite.direction
                elif player.rect.bottom in range(sprite.rect.top, sprite.rect.bottom):
                    self.drift.x = -1 / sprite.direction

    def get_soc_response(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_1]: self.SoC = 1
        if keys[pygame.K_2]: self.SoC = 2
        if keys[pygame.K_3]: self.SoC = 3
        if keys[pygame.K_4]: self.SoC = 4
        if keys[pygame.K_5]: self.SoC = 5
        if keys[pygame.K_6]: self.SoC = 6
        if keys[pygame.K_7]: self.SoC = 7
        return self.SoC

    def get_data(self, scaling):
        frame_data = pd.DataFrame(columns=self.columns)
        player = self.player.sprite
        frame_data.at[0, "player_pos"] = [player.rect.x, player.rect.y]
        frame_data.at[0, "collision"] = player.crashed
        frame_data.at[0, "current_input"] = self.current_input
        frame_data.at[0, "drift_enabled"] = self.drift_enabled
        frame_data.at[0, "current_drift"] = self.drift.x
        frame_data.at[0, "level_done"] = self.level_done
        frame_data.at[0, "input_noise_magnitude"] = self.input_noise_magnitude
        frame_data.at[0, "input_noise_on"] = self.input_noise_on
        frame_data.at[0, "time_played"] = self.time_played
        frame_data.at[0, "time_tag"] = self.eye_tracker_device.getTime() if self.eye_tracker_device else None
        frame_data.at[0, "trial"] = self.trial
        frame_data.at[0, "attempt"] = self.attempt
        frame_data.at[0, "level_size_y"] = self.level_size_y
        frame_data.at[0, "SoC"] = self.SoC
        frame_data.at[0, "visible_obstacles"] = self.visible_obstacles

        visible_drift_tiles = []
        for sprite in self.drift_tiles.sprites():
            if 0 <= sprite.rect.y <= (observation_space_size_y - bottom_edge) * scaling:
                visible_drift_tiles.append([sprite.rect.x, sprite.rect.y])
        frame_data.at[0, "visible_drift_tiles"] = visible_drift_tiles

        if self.current_gaze:
            gaze_x, gaze_y = self.current_gaze
        else:
            gaze_x, gaze_y = None, None
        frame_data.at[0, "gaze_x"] = gaze_x
        frame_data.at[0, "gaze_y"] = gaze_y

        self.data = pd.concat([self.data, frame_data], ignore_index=True)

    def run(self, time_played, player_position, scaling, tiny_visualization=False, keyboard_input=False):
        self.time_played = time_played
        player = self.player.sprite

        self.update_gaze()

        self.visible_obstacles = []
        for sprite in self.comets.sprites():
            if 0 <= sprite.rect.y <= (observation_space_size_y - bottom_edge) * scaling:
                self.visible_obstacles.append([sprite.rect.x, sprite.rect.y])

        self.adjacent_wall_tiles_x_pos = [self.walls.sprites()[0].rect.x, self.walls.sprites()[1].rect.x]

        finish_line = self.finish_line.sprites()[-1]
        if finish_line.rect.bottom < player.rect.top:
            if question_soc:
                display_soc_question(self.display_surface)
                response = self.get_soc_response()
                if response is not None:
                    display_soc_question(self.display_surface, answered=True)
                    self.level_done = True
                    self.quit = True
                    self.get_data(scaling)
                    self.data.to_csv(self._get_output_filename(), sep=",")
            else:
                self.level_done = True
                self.quit = True
                self.get_data(scaling)
                self.data.to_csv(self._get_output_filename(), sep=",")
        elif player.crashed:
            if question_soc:
                display_soc_question(self.display_surface)
                response = self.get_soc_response()
                if response is not None:
                    self.quit = True
                    self.get_data(scaling)
                    self.data.to_csv(self._get_output_filename(), sep=",")
            else:
                self.quit = True
                self.get_data(scaling)
                self.data.to_csv(self._get_output_filename(), sep=",")
        else:
            self.level_done = False
            if not tiny_visualization and keyboard_input:
                player.animate(self.current_input)
            if keyboard_input:
                self.update()
            else:
                self.player.update(player_position, scaling, keyboard_input)
            if keyboard_input and player.rect.y < player_position[1]:
                player.approach(velocity, scaling)
            if not tiny_visualization and player.rect.y >= player_position[1]:
                self.comets.update(velocity, scaling, self.horizontal_movement)
                self.walls.update(velocity, scaling, self.horizontal_movement)
                self.drift_tiles.update(velocity, scaling, self.horizontal_movement)
                self.particles.update(velocity, scaling, self.horizontal_movement)
                self.finish_line.update(velocity, scaling, self.horizontal_movement)
                self.input_noise_threshold -= 1 * scaling * velocity
            if keyboard_input:
                self.check_for_collision()
                self.check_for_drift()

            self.particles.draw(self.display_surface)
            self.comets.draw(self.display_surface)
            self.walls.draw(self.display_surface)
            self.drift_tiles.draw(self.display_surface)
            self.finish_line.draw(self.display_surface)
            self.bottom_edge.draw(self.display_surface)
            self.player.draw(self.display_surface)

            if display_keys:
                draw_rect_alpha(self.display_surface, (124, 252, 0, self.transparency_right), (160, 60, 90, 90))
                draw_polygon_alpha(self.display_surface, (255, 255, 255, self.transparency_right), [(240, 105), (170, 70), (170, 140)])
                draw_rect_alpha(self.display_surface, (124, 252, 0, self.transparency_left), (60, 60, 90, 90))
                draw_polygon_alpha(self.display_surface, (255, 255, 255, self.transparency_left), [(70, 105), (140, 70), (140, 140)])

            self.get_data(scaling)

        return self.quit, self.level_done