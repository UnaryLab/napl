import torch
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 8})

x = torch.tensor([
    1.0000, 1.85355339, 1.5000, 0.85355339,
    1.0000, 1.14644661, 0.5000, 0.14644661,
    1.0000, 1.85355339, 1.5000, 0.85355339,
    1.0000, 1.14644661, 0.5000, 0.14644661
])

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

# Restore natural order from the bit-reversed FFT output.
n = 16
duration = 1  # seconds

freqs = np.fft.fftfreq(n, d=1/n)
sorted_indices = np.argsort(freqs)
freqs_plot = freqs[sorted_indices]

unary_real = unary_real[sorted_indices]
unary_imag = unary_imag[sorted_indices]
torch_real = torch_real[sorted_indices]
torch_imag = torch_imag[sorted_indices]

unary_mag = torch.sqrt(unary_real**2 + unary_imag**2)
torch_mag = torch.sqrt(torch_real**2 + torch_imag**2)

plt.figure(figsize=(3.833, 1.5))

plt.subplot(1, 2, 1)
t = np.linspace(0, duration, int(n * duration), endpoint=False)
plt.plot(t, x.tolist(), 'k-o')
plt.xticks(t[::4])
plt.xlabel('Sample index')
plt.ylabel('Amplitude')

plt.subplot(1, 2, 2)
unary_stem = plt.stem(freqs_plot, unary_mag.tolist(), linefmt='C1-', markerfmt='C1s', basefmt=' ', label='Unary FFT')
binary_stem = plt.stem(freqs_plot, torch_mag.tolist(), linefmt='C0--', markerfmt='C0o', basefmt=' ', label='Binary FFT')
plt.xlabel('Frequency bin')
plt.ylabel('Magnitude')
plt.xticks(freqs_plot[::4])
plt.legend(loc='best', fontsize=8)

binary_stem.markerline.set_markersize(4)
unary_stem.markerline.set_markersize(6)

script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, 'fft_comparison.pdf')
plt.tight_layout()
plt.savefig(save_path, dpi=300, bbox_inches='tight')
