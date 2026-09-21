import pandas as pd
import numpy as np
import glob
import os
import ast

def preprocess_eye_data(df, sample_rate=200, lambda_vel=6,
                        min_saccade_dur=0.006, blink_buffer_ms=100):
   
    required = ['time_tag', 'gaze_x', 'gaze_y']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    
        df = df.copy()
        if 'time_played' in df.columns:
            df['time'] = pd.to_numeric(df['time_played'], errors='coerce')
        elif 'time_tag' in df.columns:
            if df['time_tag'].iloc[0] > 1e6:
                df['time'] = df['time_tag'] / 1_000_000.0
            else:
                df['time'] = df['time_tag']
        else:
            raise ValueError("No time column found (time_played or time_tag)")

    n = len(df)
    dt = 1.0 / sample_rate

    
    is_missing = df['gaze_x'].isna() | df['gaze_y'].isna()
    is_missing |= ((df['gaze_x'] == 0) & (df['gaze_y'] == 0))
    is_blink = is_missing

    buffer_samples = int(blink_buffer_ms * sample_rate / 1000)
    blink_excluded = np.zeros(n, dtype=bool)
    blink_idx = np.where(is_blink)[0]
    for idx in blink_idx:
        start = max(0, idx - buffer_samples)
        end = min(n, idx + buffer_samples + 1)
        blink_excluded[start:end] = True

    valid_for_vel = ~blink_excluded

   
    x_interp = df['gaze_x'].copy().astype(float)
    y_interp = df['gaze_y'].copy().astype(float)
    x_interp[(x_interp == 0) & (~is_blink)] = np.nan
    y_interp[(y_interp == 0) & (~is_blink)] = np.nan
    max_gap = int(0.2 * sample_rate)  
    x_interp = x_interp.interpolate(method='linear', limit=max_gap, limit_direction='both')
    y_interp = y_interp.interpolate(method='linear', limit=max_gap, limit_direction='both')
    vx = np.gradient(x_interp, dt)
    vy = np.gradient(y_interp, dt)
    vel = np.sqrt(vx**2 + vy**2)

    median_vel = np.median(vel[valid_for_vel])
    vel_thresh = median_vel * lambda_vel
    is_saccade_candidate = (vel > vel_thresh) & valid_for_vel

    saccade_events = []
    in_sac = False
    start_i = None
    for i in range(n):
        if is_saccade_candidate[i] and not in_sac:
            in_sac = True
            start_i = i
        elif not is_saccade_candidate[i] and in_sac:
            in_sac = False
            end_i = i - 1
            if start_i is None:
                continue
            dur = (end_i - start_i + 1) * dt
            if dur >= min_saccade_dur:
                saccade_events.append({'start_idx': start_i, 'end_idx': end_i,
                                       'start_time': df.iloc[start_i]['time'],
                                       'end_time': df.iloc[end_i]['time'],
                                       'duration': dur})
    if in_sac:
        end_i = n - 1
        if start_i is not None:
            dur = (end_i - start_i + 1) * dt
            if dur >= min_saccade_dur:
                saccade_events.append({'start_idx': start_i, 'end_idx': end_i,
                                       'start_time': df.iloc[start_i]['time'],
                                       'end_time': df.iloc[end_i]['time'],
                                       'duration': dur})

    is_saccade = np.zeros(n, dtype=bool)
    for s in saccade_events:
        is_saccade[s['start_idx']:s['end_idx']+1] = True

    fixation_events = []
    in_fix = False
    start_fix = None
    for i in range(n):
        is_good = (not is_saccade[i]) and (not blink_excluded[i])
        if is_good and not in_fix:
            in_fix = True
            start_fix = i
        elif (not is_good) and in_fix:
            in_fix = False
            end_fix = i - 1
            if start_fix is None:
                continue
            dur = (end_fix - start_fix + 1) * dt
            if dur > 0.02:  
                meta = {
                    'start_time': df.iloc[start_fix]['time'],
                    'end_time': df.iloc[end_fix]['time'],
                    'duration': dur,
                    'trial': df.iloc[start_fix].get('trial', None),
                    'attempt': df.iloc[start_fix].get('attempt', None),
                    'input_noise': df.iloc[start_fix].get('input_noise_magnitude', None),
                    'source_file': None
                }
                fixation_events.append(meta)
    if in_fix:
        end_fix = n - 1
        if start_fix is not None:
            dur = (end_fix - start_fix + 1) * dt
            if dur > 0.02:
                meta = {
                    'start_time': df.iloc[start_fix]['time'],
                    'end_time': df.iloc[end_fix]['time'],
                    'duration': dur,
                    'trial': df.iloc[start_fix].get('trial', None),
                    'attempt': df.iloc[start_fix].get('attempt', None),
                    'input_noise': df.iloc[start_fix].get('input_noise_magnitude', None),
                    'source_file': None
                }
                fixation_events.append(meta)

    df['is_blink_excluded'] = blink_excluded
    df['is_saccade'] = is_saccade
    df['is_fixation'] = (~blink_excluded) & (~is_saccade)

    return df, fixation_events, saccade_events


if __name__ == "__main__":
    
    data_folder = "eye_data"
    
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
  
    csv_files = [f for f in csv_files if "_processed" not in f and "all_fixations" not in f]

    all_fixations = []   

    for file_path in csv_files:
        print(f"Processing {file_path} ...")
        try:
            df = pd.read_csv(file_path)
           
            if 'gaze_x' not in df.columns or 'gaze_y' not in df.columns:
                print(f"  Skipping {file_path}: no gaze_x/gaze_y columns.")
                continue

            
            if 'time_played' in df.columns:
                times = pd.to_numeric(df['time_played'], errors='coerce').values
            elif 'time_tag' in df.columns:
                times = pd.to_numeric(df['time_tag'], errors='coerce').values
                if len(times) > 0 and times[0] > 1e6:
                    times = times / 1_000_000.0
            else:
                times = None

            if times is not None and len(times) > 0:
                diffs = np.diff(times)
                median_diff = np.median(diffs[diffs > 0])
                sample_rate = int(round(1.0 / median_diff)) if median_diff > 0 else 200
                print(f"  Detected sample rate: {sample_rate} Hz")
            else:
                sample_rate = 200  

            processed_df, fixations, saccades = preprocess_eye_data(
                df, sample_rate=sample_rate, lambda_vel=6,
                min_saccade_dur=0.006, blink_buffer_ms=100
            )

            for fix in fixations:
                fix['source_file'] = os.path.basename(file_path)

            all_fixations.extend(fixations)

            out_file = file_path.replace('.csv', '_processed.csv')
            processed_df.to_csv(out_file, index=False)
            print(f"  Saved processed data to {out_file}")
            print(f"  Found {len(fixations)} fixations.")

        except Exception as e:
            print(f"  Error processing {file_path}: {e}")

    if all_fixations:
        fix_df = pd.DataFrame(all_fixations)
        fix_df.to_csv(os.path.join(data_folder, "all_fixations.csv"), index=False)
        print(f"\nTotal fixations collected: {len(all_fixations)}")
        print("Saved to data/all_fixations.csv")
    else:
        print("\nNo fixations were found in any file.")