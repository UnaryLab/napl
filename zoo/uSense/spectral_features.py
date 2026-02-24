# loads the FFT features from the JSONL files, groups them by texture/speed/force, and saves as .npz for later use in training/testing

import itertools
import os
import re
import json
import numpy as np
from colorama import Fore, init as colorama_init
import itertools
colorama_init(autoreset=True)

# --- Configuration ---
SAMPLING_RATE = 338  # Hz
SEGMENT_LENGTH = 90  # The number of time steps in each final feature vector (9 taxels x 10 segments of 32 samples each = 90)

imbalanced_files = []

def create_grouped_features(feature_dir, output_path):
    
    if not os.path.exists(feature_dir):
        print(f"{Fore.RED}Error: Feature directory not found at {feature_dir}")
        return
    else:
        print(f"{Fore.BLUE}Processing feature directory: {feature_dir}")
    all_files = sorted([f for f in os.listdir(feature_dir) 
                    if re.search(r"_F\d+\.\d+\.jsonl$", f)])
    #print(f"Found {len(all_files)} feature files in directory: {feature_dir}")
    # file naming example: "features_T35_S6000_F1.0.jsonl"
     
    class_data = []
    class_labels = []
    class_groups = []
    class_forces = []
    class_speeds = []

    for filename in all_files:
        filepath = os.path.join(feature_dir, filename)
        if not os.path.isfile(filepath):
            print(f"{Fore.RED}Warning: {filename} is not a file. Skipping.")
            continue
        speed_val = int(next(iter(re.findall(r'(?i)(?<=_s)\d+(?=_|\.|$)', filename)), -1))
        force_val = float(next(iter(re.findall(r'(?i)(?<=_f)\d+(?:\.\d+)?(?=_|\.|$)', filename)), 'nan'))
        texture = re.search(r'(?i)(?<=features_)[^_]+', filename).group(0)
        group_id = f"{texture}_S{speed_val}_F{force_val}"
        #print(f"\nProcessing file: {filename} | Texture: {texture} | Speed: {speed_val} | Force: {force_val} | Group ID: {group_id}")
        
        filepath = os.path.join(feature_dir, filename)

        file_magnitude = [] # (9000, 3)
        try:
            with open(filepath, 'r') as f: 
                for line in f:
                    record = json.loads(line.strip())
                    mag = np.sqrt(np.array(record['real'])**2 + np.array(record['imag'])**2)
                    freqs = np.fft.fftfreq(mag.shape[0], d=1/SAMPLING_RATE)
                    freq_mask = (freqs > 0) & (freqs <= 35)
                    file_magnitude.append(mag[freq_mask])
        except (json.JSONDecodeError, KeyError) as e:
            print(f"{Fore.RED}Error reading {filename}: {e}. Skipping file.")
            continue

        # Now segment the data from this single file
        start_idx = 0
        num_segments = 0
        while start_idx + SEGMENT_LENGTH <= len(file_magnitude):
            end_idx = start_idx + SEGMENT_LENGTH
            segment = file_magnitude[start_idx:end_idx] # 1 trials = (90,3), 9 taxels, each 320 sample, i.e. 10 vectors of 32 sample
            class_data.append(segment)
            class_groups.append(group_id)
            class_forces.append(force_val)   # <- numeric
            class_speeds.append(speed_val)   # <- numeric
            class_labels.append(texture)     # <- string
            start_idx += SEGMENT_LENGTH
            num_segments += 1
        #print(f"Processed '{filename}': Class '{texture}', Group ID {group_id}, Created {num_segments} segments.")
    #print(f' data shape array {np.array(class_data).shape}')

    class_data = np.array(class_data)
    class_data = class_data.reshape(class_data.shape[0], -1)
    class_groups = np.array(class_groups)
    class_forces = np.array(class_forces)
    class_speeds = np.array(class_speeds)
    class_labels = np.array(class_labels)
    np.savez_compressed(output_path,
                        features=class_data,
                        groups=class_groups,
                        forces=class_forces,
                        speeds=class_speeds,
                        labels=class_labels)
    print(f"{Fore.GREEN}Successfully saved grouped features  of shape {class_data.shape} to:\n{output_path}")

if __name__ == "__main__":
    main_path = r"C:\unary"
    # num_variable_layers = 4  # Layers 1 through 4
    # bitwidth_options = [2, 3, 4, 5, 6]
    # last_layer_bitwidth = 3
    # variable_layer_combos = itertools.product(bitwidth_options, repeat=num_variable_layers)
    # all_combinations = [combo + (last_layer_bitwidth,) for combo in variable_layer_combos]

    values = [2, 3, 4, 5, 6]  # allowed bitwidths
    num_layers = 5

    all_combinations = []
    for layer_index in range(num_layers):
        for v in values:
            config = [6] * num_layers  # default fixed value (example)
            config[layer_index] = v    # tune one layer only
            all_combinations.append(tuple(config))

    # values = [2, 3, 4, 5]
    # all_combinations = [
    #     (*combo, 3)           # layer 5 fixed to 3
    #     for combo in itertools.product(values, repeat=4)
    #]

    # l1_vals = [2, 3, 4, 5, 6]
    # l5_vals = [2, 3, 4, 5, 6] 
    # all_combinations = [(l1, 3, 3, 3, l5) for l1 in l1_vals for l5 in l5_vals]

    # all_combinations = []
    # for b in [2,3,4,5,6,7,8]:
    #     all_combinations.append((b,b,b,b,b))
    
    # l1_vals = [2, 3, 4, 5, 6]
    # l5_vals = [2, 3, 4, 5, 6] 
    # all_combinations = [(l1, 3, 3, 3, l5) for l1 in l1_vals for l5 in l5_vals]

    print(f"Total configurations to test: {len(all_combinations)}")
    for idx, combo in enumerate(all_combinations):   
        print(f"configuration {idx + 1}/{len(all_combinations)}: {combo}")
        path =  os.path.join(main_path, f"{combo}")
        if not os.path.exists(path):
            print(f"{Fore.RED}Directory not found: {path}, skipping...")
            continue
        SAVE_PATH = os.path.join(path, "features.npz")
        if os.path.exists(SAVE_PATH): 
            print(f"{Fore.YELLOW}Skipping existing file for config: {combo}")
            continue
        print(f"configuration {idx + 1}/{len(all_combinations)}: {combo}")
        create_grouped_features(path, SAVE_PATH)
