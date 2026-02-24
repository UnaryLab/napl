
import os
import numpy as np
import pandas as pd


user = os.getenv('USERNAME') or os.getenv('USER') or 'user'
def load_texture_data(data_root, bump_type, bump_density, speed, force, trial, save_numpy=True):
    folder_name = f"Texture0{bump_type}{bump_density}"
    file_name = f"Data[Texture(0{bump_type}{bump_density})][Speed({speed})][Force({force})][Trial({trial})].csv"
    full_path = fr"C:\tactile\Data\{folder_name}\{file_name}"
    if not os.path.exists(full_path):
        return None

    data = pd.read_csv(full_path, header=None).values

    if save_numpy:
        out_dir = os.path.join(data_root, folder_name)
        os.makedirs(out_dir, exist_ok=True)  
        save_path = os.path.join(out_dir, f"processed_S{speed}_F{force}_T{trial}.npz")
        np.savez_compressed(save_path, data=data)
        return save_path
    else:
        return data


data_root = fr"C:\tactile"
# Select configuration
bump_types = [0, 1, 2, 3]
bump_densities = [0, 1, 2, 3, 4, 5]
speeds = [1200, 2400, 3600, 4800, 6000]
forces = [0.0, 0.5, 1.0, 1.5]
trials = list(range(1, 101))

for bump_type in bump_types:
    for bump_density in bump_densities:
        print(f"Processing Texture0{bump_type}{bump_density}...")
        for s in speeds:
            for f in forces:
                print(f"  Processing speed: {s}, force: {f}")
                for t in trials:
                    npz_path = load_texture_data(data_root, bump_type, bump_density, s, f, t)
                    if npz_path is not None:
                        print(f"Processed: Speed={s}, Force={f}, Trial={t} -> {npz_path}")
print("All processing complete.")


# ==== Config ====
TYPES = [0  , 1, 2, 3]
DENSITIES = [1, 2, 3, 4, 5]

SPEEDS = [1200, 2400, 3600, 4800, 6000] 
FORCES = [0.0, 0.5, 1.0, 1.5]
SAMPLING_RATE = 338
SPEED_TIME_RANGES = {1200: (18.5, 19.5),
                    2400: (18.0, 19.0),
                    3600: (17.8, 18.8),
                    4800: (17.7, 18.7),
                    6000: (17.5, 18.5),
                }

texture_ids = [f"{t}{d}" for t in TYPES for d in DENSITIES]
# add texture 0 with density 0
texture_ids.append("00")
print(f' total texture ids: {len(texture_ids)} -> {texture_ids}')

# # ==== Functions ====

def load_signal(file_path, speed):
    try:
        with np.load(file_path, allow_pickle=True) as data:
            signal = data['data'][:, :9]
        start_sec = SPEED_TIME_RANGES[speed][0]
        end_idx = int((start_sec + 1) * SAMPLING_RATE)
        start_idx = int(start_sec * SAMPLING_RATE)
        signal = signal[start_idx:end_idx, :] #  (338, 9)
        signal = np.array(signal, dtype=np.float32) 
        return signal
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


# ==== Main Execution ====
BASE_PATH = r'C:\tactile'  # Update this path as needed
if __name__ == "__main__":
    textures_data = []
    labels = []  
    save_dir = os.path.join(r"C:\unary", "segmented_data")
    # os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"segmented_data.npz")
    if os.path.exists(save_path):
        open(save_path, "w").close()
        print(f"Emptied {save_path}")
    else:
        print(f"Creating new file {save_path}")
    print(f"Processing textures: {texture_ids}")
    for texture in texture_ids:
        trials = 0
        check = []
        print(f"{texture} processing...")
        for speed in SPEEDS:
            for force in FORCES:
                folder = f'Texture0{texture}'
                label = f'T{texture}_S{speed}_F{force}'
                for trial in range(1, 101):
                    file = f'processed_S{speed}_F{force}_T{trial}.npz'
                    path = os.path.join(BASE_PATH, folder, file)
                    if not os.path.isfile(path) or os.path.getsize(path) == 0:
                        print(f"File not found or empty: {path}")
                        continue
                    # print(f"Loading file: {path}") 
                    signal = load_signal(path, speed) # (338, 9) for 1 second of data 
                    trials += 1
                    if signal is not None: # segment into 10 segments of 32 samples each
                        for start in range(0, signal.shape[0] - 32 + 1, 32):
                            segment = signal[start:start + 32]
                            textures_data.append(segment)
                            check.append(segment)
                            labels.append(label)
                    # each signal shape here (10, 32, 9)
                    #print(f' Loaded texture {texture}, speed {speed}, force {force}, trial {trial}, concat shape: {np.array(check).shape}, labels shape {np.array(labels).shape}')
                print(f' data shape for comb is : {np.array(check).shape}')
        print(f' Number of trials for texture {texture}: {trials}')
        print(f' shape of check: texture{texture} {np.array(check).shape}')
    if textures_data: 
        textures_data = np.array(textures_data)
        print(f' texture data shape: {textures_data.shape}')
        textures_data = textures_data.transpose(0, 2, 1)  # 
        print(f"Shape of textures_data before reshaping: {textures_data.shape}")
        textures_data = textures_data.reshape(-1, 32)      
        labels = np.repeat(labels, 9)  # Repeat labels for each segment
        print(f' shape of labels: {np.array(labels).shape}')
        labels = np.array(labels)
        print(f'Final shape of textures_data: {textures_data.shape}, labels: {labels.shape}')
        print(f'unique labels: {set(labels)}')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, data=textures_data, labels=labels)  # <-- Save both
        print(f"Saved {save_path}")
    else:
        print(f"No data found for texture {texture} with segments of length 32.")

