import time
import sys

import pygame.display
from pygame import VIDEORESIZE

from helper_functions import (
    get_drift_ranges,
    get_obstacles_lists,
    get_player_positions,
    get_wall_positions,
    adjust_player_positions,
    adjust_drift_ranges,
    adjust_obstacles_list,
    adjust_wall_list,
)
from level_setup import Level


def run_visualization(
    surface,
    scaling=1,
    tiny_visualization=False,
    FPS=30,
    eye_tracker_device=None,
    keyboard_input=False,
    obstacles_lists_file="obstacles_list.csv",
    drift_ranges_file="drift_ranges.csv",
    wall_list_file="walls_dict.csv",
    input_noise_magnitude=None,
    input_noise_threshold=0,
    drift_enabled=True,
    trial=0,
    attempt=0,
    n_run=0,
    code="test",
    gaze_queue=None,
):
    """
    :param surface: argument for specifying pygame.display object
    :param scaling: int (or float) to scale up on-screen visualization
    :param tiny_visualization: bool argument to visualize complete level in tiny
    :param FPS: frames per second. 30 as default value. For smoother animation choose 60.
    :param eye_tracker_device: DATAPixx3 device recording eye-tracking data
    :param keyboard_input: bool argument needed to specify whether vis for human experiment or simple data visualization
    :param obstacles_lists_file: file for list of obstacles. Has to be in logs repository
    :param drift_ranges_file: file for drift ranges. Has to be in logs repository
    :param wall_list_file: file with wall positions on every y position in level
    :param input_noise_magnitude: input noise imposed on player throughout level
    :param input_noise_threshold: from this point in level input noise will be applied
    :param drift_enabled: drift tiles in level vs. no drift tiles
    :param trial: player movements of which trial (.csv file in logs) to be visualized
    :param attempt: attempts for this specific trial
    :param n_run: total number of runs in this specific experiment / visualization
    :param code: individual code for participant in case of experiment
    """
    # preparing lists of in-game objects from which to draw said objects on screen
    # walls will be the same across all experimental trials
    wall_list = get_wall_positions(wall_list_file)
    wall_list = adjust_wall_list(wall_list, scaling)

    obstacles_list, flag_multiple_obstacle_lists = get_obstacles_lists(
        obstacles_lists_file
    )
    obstacles_list = adjust_obstacles_list(obstacles_list, scaling)

    if keyboard_input:
        player_positions_filename = "0_vis.csv"
    else:
        player_positions_filename = str(trial) + "_vis.csv"
    player_starting_position, player_positions = get_player_positions(
        player_positions_filename
    )
    player_starting_position, player_positions = adjust_player_positions(
        player_starting_position,
        player_positions,
        scaling,
        tiny_visualization=tiny_visualization,
    )

    drift_ranges = get_drift_ranges(drift_ranges_file)
    drift_ranges = adjust_drift_ranges(drift_ranges, scaling)

    if not keyboard_input:
        player_positions = iter(player_positions)

    # running through game loop
    if keyboard_input:
        level = Level(
            wall_list=wall_list,
            obstacles_list=obstacles_list,
            player_starting_position=player_starting_position,
            input_noise_magnitude=input_noise_magnitude,
            input_noise_threshold=input_noise_threshold,
            drift_ranges=drift_ranges,
            drift_enabled=drift_enabled,
            screen=surface,
            scaling=scaling,
            n_run=n_run,
            keyboard_input=keyboard_input,
            trial=trial,
            attempt=attempt,
            code=code,
            FPS=FPS,
            eye_tracker_device=eye_tracker_device,
            gaze_queue=gaze_queue,  
        )

        level_done = run_pygame(
            surface=surface,
            scaling=scaling,
            FPS=FPS,
            keyboard_input=keyboard_input,
            player_positions=player_positions,
            level=level,
            tiny_visualization=tiny_visualization,
        )
        return level_done
    else:
        level = Level(
            wall_list=wall_list,
            obstacles_list=obstacles_list,
            player_starting_position=player_starting_position,
            drift_ranges=drift_ranges,
            screen=surface,
            scaling=scaling,
            keyboard_input=keyboard_input,
            code=code,
            FPS=FPS,
            eye_tracker_device=eye_tracker_device
        )

        run_pygame(
            surface=surface,
            scaling=scaling,
            FPS=FPS,
            keyboard_input=keyboard_input,
            player_positions=player_positions,
            level=level,
            tiny_visualization=tiny_visualization,
        )


def run_pygame(
    surface,
    scaling,
    FPS,
    keyboard_input,
    player_positions,
    level,
    tiny_visualization,
):
    """
    :param surface: pygame surface on which objects are drawn
    :param scaling: integer or float which is used to enlarge visualization
    :param FPS: how many frames per second should be drawn
    :param keyboard_input: True when human playing vs. False when only visualization
    :param player_positions: only used in simple visualization; where player should be drawn
    :param level: level object which is defined before
    :param tiny_visualization: True vs. False; when True scaling parameter specific for tiny vis
    """
    clock = pygame.time.Clock()

    # time onset
    game_start_time = time.time()

    level_done = False

    if keyboard_input:
        quit_ = False
        while not quit_:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                # FIXME: resizing not working
                if event.type == VIDEORESIZE:
                    surface = pygame.display.set_mode(
                        (event.w, event.h), pygame.RESIZABLE
                    )

            # update time
            time_played = time.time() - game_start_time
            # print(time_played)

            surface.fill("black")

            # update player position
            current_player_position = player_positions[
                0
            ]  # not needed but still given in level.run()
            quit_, level_done = level.run(
                time_played,
                current_player_position,
                scaling,
                tiny_visualization=tiny_visualization,
                keyboard_input=keyboard_input,
            )

            pygame.display.update()
            clock.tick(FPS)
    else:
        for player_position in player_positions:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                # FIX ME: resizing not working
                if event.type == VIDEORESIZE:
                    surface = pygame.display.set_mode(
                        (event.w, event.h), pygame.RESIZABLE
                    )

            # update time
            time_played = time.time() - game_start_time

            surface.fill("black")

            # update player position
            current_player_position = player_position
            level.run(
                time_played,
                current_player_position,
                scaling,
                tiny_visualization=tiny_visualization,
                keyboard_input=keyboard_input,
            )

            pygame.display.update()
            clock.tick(FPS)

    return level_done
