
import torch
import os, sys
import numpy as np
import json
from torch.utils.data import Dataset, DataLoader
import time
from colorama import Fore, init
init(autoreset=True)
from napl.algorithm.fft import napl_fft
import itertools
import ast

combinations_to_run = ast.literal_eval(sys.argv[1])

def log(msg):
    print(time.strftime("[%Y-%m-%d %H:%M:%S] ") + str(msg), flush=True)
    
batch_size = 20000 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    log(f"{Fore.GREEN}Using GPU: {torch.cuda.get_device_name(0)}")
    torch.cuda.empty_cache()
else:
    log(f"{Fore.YELLOW}Using CPU")

# --- Dataset and DataLoader ---

path = "/content/output/segmented_data/segmented_data.npz"
SAVE_DIR = "/content/output/fft_features"

class NumpyMemmapDataset(Dataset):
    def __init__(self, npz_path):
        npz = np.load(npz_path, mmap_mode='r')
        self.data = npz["data"]      # shape: (N, 32, 9)
        self.labels = npz["labels"]  # shape: (N,)
        print(f"Loaded data shape: {self.data.shape}, labels shape: {self.labels.shape}")
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = str(self.labels[idx]) 
        return x, y

if not os.path.isfile(path):
    print(f"{Fore.RED}File not found: {path}")
    sys.exit()

print(f"Loading dataset from {path} ...")
dataset = NumpyMemmapDataset(path)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

_initialized_labels = {}
def append_feature_jsonl(complex_fft_result, label, config_name):
    
    path = os.path.join(comb_path, f"features_{label}.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if config_name not in _initialized_labels:
        _initialized_labels[config_name] = set()
    if label not in _initialized_labels[config_name]:
        if os.path.exists(path): open(path, "w").close()
        _initialized_labels[config_name].add(label)
    record = {
        "real": complex_fft_result.real.tolist(),
        "imag": complex_fft_result.imag.tolist(),
        "label": label
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

# ==============================================================================
total_batches = len(loader)
print(f"\n Starting FFT processing for {total_batches} batches...")


for idx, combo in enumerate(combinations_to_run):
   
    config_name = str(combo) 
    print(f"\nProcessing configuration {idx + 1}/{len(combinations_to_run)}: {config_name}")

    comb_path = os.path.join(SAVE_DIR, config_name)
    if os.path.exists(comb_path):
        print(f"{Fore.YELLOW}Skipping existing configuration: {config_name}")
        continue
    model = napl_fft(combo, device=device).eval()
    
    for batch_idx, (batch_x, batch_y) in enumerate(loader):
        start_time = time.time()
        config_name = combo
        batch_x = batch_x - batch_x.mean(dim=-1, keepdim=True)
        batch_x = batch_x.to(device)

        print(f"Processing batch {batch_idx + 1}/{total_batches} with shape: {batch_x.shape} and labels: {len(batch_y)}")
        if device.type == 'cuda':
            torch.cuda.synchronize()  

            start = torch.cuda.Event(enable_timing=True)
            end   = torch.cuda.Event(enable_timing=True)

            start.record()
            with torch.no_grad():
                _, unary_output = model(batch_x, verbose=False)
            end.record()

            torch.cuda.synchronize()  
            napl_time_ms = start.elapsed_time(end)  
        else:
            import time
            t0 = time.perf_counter()
            with torch.no_grad():
                _, unary_output = model(batch_x, verbose=False)
            t1 = time.perf_counter()
            napl_time_ms = (t1 - t0) * 1e3
        
        for i in range(batch_x.shape[0]):
            label = batch_y[i]
            fft_sample = unary_output[i].cpu().numpy()
            append_feature_jsonl(fft_sample, label, str(config_name))

        print(f"Batch {batch_idx + 1}/{total_batches} processed in {napl_time_ms:.4f} milliseconds.")
        model.clear_cache()

print("\nAll batches processed successfully.")

