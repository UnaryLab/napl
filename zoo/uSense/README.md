# uSense: Tactile Texture Classification Pipeline

## Overview

This directory contains the end-to-end uSense pipeline for tactile texture classification, including:

- Stochastic FFT-based feature extraction  
- Feature structuring  
- Texture classification  

Representative tactile data used in the study are publicly available [here](https://github.com/LIMB-UCF/uSense-tactile-data).

The complete dataset is available upon request.


## Repository Structure

- `fft_main_run.py`  
  Executes the stochastic FFT in uSense.

- `spectral_features.py`  
  Constructs frequency-domain magnitude features.

- `classification.py`  
  Texture classification.
