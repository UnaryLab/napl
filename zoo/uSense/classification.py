import numpy as np
import os, sys
from sklearn import svm
from colorama import Fore, init as colorama_init
import json 
from sklearn.model_selection import GroupKFold, cross_val_score
import itertools
from colorama import Fore, init as colorama_init
colorama_init(autoreset=True)


def load_data(path):
    if not os.path.exists(path):
        #print(f"{Fore.RED}File not found: {path}")
        return None, None, None
    try:
        loaded_data = np.load(path, allow_pickle=True)
        magnitude = loaded_data['features']
        labels = loaded_data['labels']
        groups = loaded_data.get('groups') if 'groups' in loaded_data.files else None
        return magnitude, labels, groups
    except Exception as e:
        #print(f"{Fore.MAGENTA}Error loading data from {path}: {e}")
        return None, None, None

if __name__ == "__main__":
    user = os.getlogin()
    save_path = r"C:\unary"
    if not os.path.exists(save_path):
        print(f"{Fore.RED}Directory not found: {save_path}")
        sys.exit()
    save_file = os.path.join(save_path, "classification_results.jsonl")
  
    existing_configs = set()
    if os.path.exists(save_file):
        print(f"{Fore.YELLOW}Results file found. Reading existing configurations to avoid re-computation.")
        with open(save_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if 'configuration' in data:
                        existing_configs.add(tuple(data['configuration']))
                except json.JSONDecodeError:
                    print(f"{Fore.RED}Warning: Skipping a malformed line in {save_file}")
        #print(f"Found {len(existing_configs)} previously completed configurations.")
    else:
        print(f"{Fore.YELLOW}Creating new file: {save_file}")

    # num_variable_layers = 4 
    # bitwidth_options = [2, 3, 4, 5, 6]
    # last_layer_bitwidth = 3
    # variable_layer_combos = itertools.product(bitwidth_options, repeat=num_variable_layers)
    # all_combinations = [combo + (last_layer_bitwidth,) for combo in variable_layer_combos]

    # l1_vals = [2, 3, 4, 5, 6]
    # l5_vals = [2, 3, 4, 5, 6]
    # all_combinations = [(l1, 3, 3, 3, l5) for l1 in l1_vals for l5 in l5_vals]
    
    # values = [2, 3, 4, 5]
    # all_combinations = [
    #     (*combo, 3)           # layer 5 fixed to 3
    #     for combo in itertools.product(values, repeat=4)
    # ]

    # all_combinations = [] #initial attempt
    # for b in [2,3,4,5,6,7,8]:
    #     all_combinations.append((b,b,b,b,b))
    # print(f"Total configurations to test: {len(all_combinations)}")

    all_combinations = [] #TUNING
    values = [2, 3, 4, 5]  # allowed bitwidths
    num_layers = 5
    for layer_index in range(num_layers):
        for v in values:
            config = [6] * num_layers  # default fixed value (example)
            config[layer_index] = v    # tune one layer only
            all_combinations.append(tuple(config))

    for idx, combo in enumerate(all_combinations):
        if combo in existing_configs:
            print(f"{Fore.CYAN}Skipping configuration {combo} as it already exists in the results file.")
            continue
        print(f"Testing configuration {idx + 1}/{len(all_combinations)}: {combo}")
        #path =  os.path.join(save_path, f"{combo}\segmented_features_grouped.npz")
        path = os.path.join(save_path, str(combo), "features.npz")
        if not os.path.exists(path):
            print(f"{Fore.RED} path: {path} Data file not found for configuration {combo}, skipping...")
            continue
        magnitude, labels, groups = load_data(path)
        if magnitude is None or labels is None or groups is None:
            print(f"{Fore.RED}Skipping configuration {combo} due to data loading issues.")
            continue
        magnitude_flat = magnitude.reshape(magnitude.shape[0], -1)
        
        clf = svm.SVC()
        gkf = GroupKFold(n_splits=5)
        cv_scores = cross_val_score(clf, magnitude_flat, labels, cv=gkf, groups=groups)
        print(f"{Fore.GREEN}Cross-validated Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
   
        results = {
            "configuration": list(combo),  
            "accuracy": cv_scores.mean(),
            "std_dev": cv_scores.std(),
            "all_scores": cv_scores.tolist() 
        }
        print(f"{Fore.BLUE}Results for configuration {combo}: {results}")

        with open(save_file, "a") as f:
            f.write(json.dumps(results) + "\n")
        print(f"{Fore.GREEN}Results saved for configuration {combo}.")

