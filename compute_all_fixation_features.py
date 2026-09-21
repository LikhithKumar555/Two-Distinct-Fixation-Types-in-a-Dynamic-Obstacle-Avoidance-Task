import pandas as pd
import numpy as np
import ast
import os
from scipy.spatial.distance import cdist


DATA_FOLDER = "eye_data"
SCREEN_WIDTH_PX = 1920      
SCREEN_HEIGHT_PX = 1080
VIEWING_DISTANCE_CM = 80    
SCREEN_WIDTH_CM = 62        

SPACESHIP_PX = (954, 270)


visual_angle_width = 2 * np.arctan((SCREEN_WIDTH_CM / 2) / VIEWING_DISTANCE_CM) * (180 / np.pi)
deg_per_pixel = visual_angle_width / SCREEN_WIDTH_PX

fix_df = pd.read_csv(os.path.join(DATA_FOLDER, "all_fixations.csv"))
print(f"Loaded {len(fix_df)} fixations")

dist_to_spaceship_deg = []
dist_to_closest_obstacle_deg = []


for idx, row in fix_df.iterrows():
    source_file = row['source_file']   
    raw_csv_path = os.path.join(DATA_FOLDER, source_file)
    
    
    try:
        raw_df = pd.read_csv(raw_csv_path)
       
        if 'time_played' in raw_df.columns:
            raw_df['time_sec'] = pd.to_numeric(raw_df['time_played'], errors='coerce')
        elif 'time_tag' in raw_df.columns:
            if raw_df['time_tag'].iloc[0] > 1e6:
                raw_df['time_sec'] = raw_df['time_tag'] / 1_000_000.0
            else:
                raw_df['time_sec'] = raw_df['time_tag']
        elif 'time' in raw_df.columns:
            raw_df['time_sec'] = raw_df['time']
        else:
            raise ValueError("No time column (time_played, time_tag, or time)")
        
        mask = (raw_df['time_sec'] >= row['start_time']) & (raw_df['time_sec'] <= row['end_time'])
        interval_df = raw_df[mask]
        if len(interval_df) == 0:
            dist_to_spaceship_deg.append(np.nan)
            dist_to_closest_obstacle_deg.append(np.nan)
            continue
        
        mean_gaze_x = interval_df['gaze_x'].mean()
        mean_gaze_y = interval_df['gaze_y'].mean()
    
        gaze_px_x = mean_gaze_x * SCREEN_WIDTH_PX
        gaze_px_y = mean_gaze_y * SCREEN_HEIGHT_PX
        
        dist_px = np.hypot(gaze_px_x - SPACESHIP_PX[0], gaze_px_y - SPACESHIP_PX[1])
        dist_deg = dist_px * deg_per_pixel
        dist_to_spaceship_deg.append(dist_deg)
        
    
        min_obs_deg = np.inf
      
        for _, frame in interval_df.iterrows():
            obs_str = frame.get('visible_obstacles', '[]')
            if pd.isna(obs_str) or obs_str == '' or obs_str == '[]':
                continue
            try:
                obstacles = ast.literal_eval(obs_str)
            except:
                continue
           
            for obs in obstacles:
                dist_px_obs = np.hypot(gaze_px_x - obs[0], gaze_px_y - obs[1])
                dist_deg_obs = dist_px_obs * deg_per_pixel
                if dist_deg_obs < min_obs_deg:
                    min_obs_deg = dist_deg_obs
        if min_obs_deg == np.inf:
            min_obs_deg = np.nan
        dist_to_closest_obstacle_deg.append(min_obs_deg)
        
    except Exception as e:
        print(f"Error processing fixation {idx} from {source_file}: {e}")
        dist_to_spaceship_deg.append(np.nan)
        dist_to_closest_obstacle_deg.append(np.nan)

fix_df['dist_to_spaceship_deg'] = dist_to_spaceship_deg
fix_df['Dist_to_closest_obstacles_deg'] = dist_to_closest_obstacle_deg

fix_df['N_visible_obstacles'] = np.nan  


output_path = os.path.join(DATA_FOLDER, "all_fixations_with_features.csv")
fix_df.to_csv(output_path, index=False)
print(f"Saved {len(fix_df)} fixations with features to {output_path}")
print(f"Columns: {fix_df.columns.tolist()}")