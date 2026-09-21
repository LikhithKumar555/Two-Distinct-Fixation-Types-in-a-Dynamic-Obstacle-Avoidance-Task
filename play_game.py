import pygame
import sys
import os
import random
import time
from level_setup import Level
from helper_functions import (
    get_wall_positions, get_obstacles_lists, get_player_positions, get_drift_ranges,
    adjust_wall_list, adjust_obstacles_list, adjust_player_positions, adjust_drift_ranges
)
from config import scaling, observation_space_size_x, observation_space_size_y, edge
from pupil_core_interface import PupilCore

PARTICIPANT_CODE = input("Enter participant code: ")
os.makedirs("data", exist_ok=True)
FPS = 60
MAX_ATTEMPTS_PER_COMBO = 3

LAYOUTS = list(range(1, 7))

NOISE_LEVELS = [None, "weak", "medium", "strong", "very_strong"]

all_combinations = [(layout, noise) for layout in LAYOUTS for noise in NOISE_LEVELS]
random.shuffle(all_combinations)   

attempt_counter = {}

pygame.init()
screen_width = (observation_space_size_x + (2 * edge)) * scaling
screen_height = observation_space_size_y * scaling
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption(f"Gaze Experiment – {PARTICIPANT_CODE}")
pygame.mouse.set_visible(False)

print("Connecting to Pupil Core...")
pupil = PupilCore(ip='127.0.0.1', port=50020)
pupil.start_gaze_listener()
print("Gaze listener started. (Make sure Pupil Capture is running and already calibrated)")

def show_start_instruction():
    font = pygame.font.SysFont("Arial", 48)
    text = font.render("Press SPACE to start experiment", True, (255,255,255))
    screen.fill((0,0,0))
    screen.blit(text, (screen_width//2 - text.get_width()//2,
                       screen_height//2 - text.get_height()//2))
    pygame.display.flip()
    print("Waiting for SPACE...")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pupil.close()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return

def show_intertrial_screen(crashed):
    font = pygame.font.SysFont("Arial", 48)
    if crashed:
        line1 = font.render("You crashed!", True, (255,100,100))
        line2 = font.render("Press SPACE for next trial", True, (255,255,255))
    else:
        line1 = font.render("Trial completed!", True, (100,255,100))
        line2 = font.render("Press SPACE for next trial", True, (255,255,255))
    screen.fill((0,0,0))
    screen.blit(line1, (screen_width//2 - line1.get_width()//2, screen_height//2 - 60))
    screen.blit(line2, (screen_width//2 - line2.get_width()//2, screen_height//2 + 20))
    pygame.display.flip()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pupil.close()
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return

def run_trial(layout_num, noise_level, attempt):
    print(f"\n--- Layout {layout_num}, noise={noise_level}, attempt {attempt} ---")

    
    try:
        wall_list = get_wall_positions(f"walls_dict_{layout_num}.csv")
        obstacles_list, _ = get_obstacles_lists(f"object_list_{layout_num}.csv")
        player_starting_position, player_positions = get_player_positions("0_vis.csv")
        drift_ranges = get_drift_ranges(f"drift_ranges_{layout_num}.csv")
    except Exception as e:
        print(f"ERROR loading files for layout {layout_num}: {e}")
        return False

    wall_list = adjust_wall_list(wall_list, scaling)
    obstacles_list = adjust_obstacles_list(obstacles_list, scaling)
    player_starting_position, player_positions = adjust_player_positions(
        player_starting_position, player_positions, scaling
    )
    drift_ranges = adjust_drift_ranges(drift_ranges, scaling)

    current_player_position = player_positions[0] if player_positions else [50, 10]

    
    desired_csv = f"data/experiment_{PARTICIPANT_CODE}_layout{layout_num}_{noise_level}_attempt{attempt}.csv"

    level = Level(
        wall_list=wall_list,
        obstacles_list=obstacles_list,
        player_starting_position=player_starting_position,
        drift_ranges=drift_ranges,
        screen=screen,
        scaling=scaling,
        code=PARTICIPANT_CODE,
        keyboard_input=True,
        trial=layout_num,
        attempt=attempt,
        n_run=attempt,
        drift_enabled=True,
        input_noise_magnitude=noise_level,
        input_noise_threshold=1500,
        gaze_queue=pupil.gaze_queue,
        gaze_output_file=desired_csv,    
    )

    
    clock = pygame.time.Clock()
    game_start_time = pygame.time.get_ticks() / 1000.0
    quit_flag = False
    level_done = False

    while not quit_flag and not level_done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quit_flag = True
        time_played = pygame.time.get_ticks() / 1000.0 - game_start_time
        screen.fill("black")
        quit_flag, level_done = level.run(
            time_played,
            current_player_position,
            scaling,
            keyboard_input=True,
        )
        pygame.display.update()
        clock.tick(FPS)

    if os.path.exists(desired_csv):
        print(f"Saved: {desired_csv}")
    else:
        print(f"Warning: {desired_csv} not found – gaze data may not have been saved.")

    return level_done

show_start_instruction()

combinations_remaining = all_combinations.copy()

for (layout_num, noise_level) in combinations_remaining:
    success = False
    attempt = 1
    while not success and attempt <= MAX_ATTEMPTS_PER_COMBO:
        success = run_trial(layout_num, noise_level, attempt)
        if not success:
            attempt += 1
            if attempt <= MAX_ATTEMPTS_PER_COMBO:
                print(f"Retrying layout {layout_num}, noise={noise_level} (attempt {attempt})...")
                time.sleep(1)

    if not success:
        print(f"Skipping layout {layout_num}, noise={noise_level} after {MAX_ATTEMPTS_PER_COMBO} attempts.")

    if combinations_remaining.index((layout_num, noise_level)) < len(combinations_remaining) - 1:
        show_intertrial_screen(not success)

pupil.close()
pygame.quit()
print("\nExperiment finished. All CSVs are in the 'data' folder.")