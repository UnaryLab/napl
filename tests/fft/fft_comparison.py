import torch
import os
import numpy as np
import matplotlib.pyplot as plt

# Set global font size to 8
plt.rcParams.update({'font.size': 8})

# Input signal (time domain)
x = torch.tensor([
    1.0000, 1.85355339, 1.5000, 0.85355339,
    1.0000, 1.14644661, 0.5000, 0.14644661,
    1.0000, 1.85355339, 1.5000, 0.85355339,
    1.0000, 1.14644661, 0.5000, 0.14644661
])

# Unary FFT output (from user)
unary_real = torch.tensor([
    15.9375, -0.0391, -0.0234, -0.0234,
    -0.0156, 0.0000, 0.0000, -0.0156,
    0.0000, 0.0000, 0.0078, -0.0078,
    -0.0078, 0.0000, 0.0000, -0.0078
])
unary_imag = torch.tensor([
    -0.0078, 0.0000, -4.0078, -0.0156,
    -4.0000, 0.0000, -0.0234, -0.0078,
    -0.0078, 0.0000, 0.0000, 0.0000,
    3.9922, 0.0000, 3.9922, 0.0000
])

# Torch FFT result (ground truth)
torch_real = torch.tensor([
    16.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0
])
torch_imag = torch.tensor([
    0.0, 0.0, -4.0, 0.0,
    -4.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0,
    4.0, 0.0, 4.0, 0.0
])

# bit reversed order
# The FFT output is in bit-reversed order, so we need to sort it
n = 16
# duration in seconds
duration = 1

freqs = np.fft.fftfreq(n, d=1/n)
sorted_indices = np.argsort(freqs)
freqs_plot = freqs[sorted_indices]

unary_real = unary_real[sorted_indices]
unary_imag = unary_imag[sorted_indices]
torch_real = torch_real[sorted_indices]
torch_imag = torch_imag[sorted_indices]

# Compute magnitudes using PyTorch
unary_mag = torch.sqrt(unary_real**2 + unary_imag**2)
torch_mag = torch.sqrt(torch_real**2 + torch_imag**2)

# Plotting in inch
# Set figure size in inches, in points
# 1 inch = 72 points, so 3.833 inches = 3.833 * 72 points
# 3.833 inches = 275 points
plt.figure(figsize=(3.833, 1.5))

# Time-domain input
plt.subplot(1, 2, 1)
t = np.linspace(0, duration, int(n * duration), endpoint=False)
plt.plot(t, x.tolist(), 'k-o')
# plt.title("Input Signal (Time Domain)")
plt.xticks(t[::4])
plt.xlabel("Sample index")
plt.ylabel("Amplitude")

# Frequency-domain comparison
plt.subplot(1, 2, 2)
unary_stem = plt.stem(freqs_plot, unary_mag.tolist(), linefmt='C1-', markerfmt='C1s', basefmt=" ", label='Unary FFT')
binary_stem = plt.stem(freqs_plot, torch_mag.tolist(), linefmt='C0--', markerfmt='C0o', basefmt=" ", label='Binary FFT')
# plt.title("Magnitude Spectrum Comparison")
plt.xlabel("Frequency bin")
plt.ylabel("Magnitude")
plt.xticks(freqs_plot[::4])
plt.legend(loc='best', fontsize=8)

# Set marker size for Unary FFT
binary_stem.markerline.set_markersize(4)
unary_stem.markerline.set_markersize(6)

# Save and display
# Get directory where the current script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, "fft_comparison.pdf")
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')

