
import math
import torch
import torch.nn as nn
from napl.utils import *
from napl.algorithm.fft.butterfly import butterfly_binary, butterfly_spike


class napl_fft(nn.Module):
    """
    napl-based Cooley-Tukey DIT FFT implementation as a PyTorch module.
    This module implements both binary and unary (spike-based) FFT computations
    using NAPL butterfly units.
    """
    verbose = False
    def __init__(self, combo = None, device = None, codec_config=None, mul_config=None, add_config=None, acc_config=None):
        """
        Initialize the NAPL FFT module.

        """

        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.combo = combo
        self.codec_config = codec_config or {
            'polarity': 'bipolar',
            'timestep': 64, # defualt setup unless specified (is updated each stage)
            'generator': 'sobol',
        }
        
        width = int(math.log2(self.codec_config['timestep']))
        self.mul_config = mul_config or {
            'polarity': 'bipolar',
            'timestep': 64,
            'generator': 'sobol',
        }
        
        self.add_config = add_config or {
            'polarity': 'bipolar',
            'scale': 1,
            'width': width + 1,
        }
        self.scale = self.add_config['scale']

        self.acc_config = acc_config or self.codec_config 
        
        self.fft_size = 32
        
        # Initialize butterfly units
        self.butterfly_binary = butterfly_binary()
        self.butterfly_spike = butterfly_spike(
            self.codec_config, self.mul_config, self.add_config, self.acc_config
        ).to(self.device)
        
        # Cache for bit-reversal indices and twiddle factors
        self._bit_reversal_cache = {}
        self._twiddle_cache = {}
    
    def _get_bit_reversal_indices(self, N):
        cache_key = (N, str(self.device))
        if cache_key not in self._bit_reversal_cache:
            num_bits = int(math.log2(N))
            indices = torch.arange(N, device=self.device)
            bit_reversed_indices = torch.zeros_like(indices)
            for i in range(num_bits):
                bit_reversed_indices |= ((indices >> i) & 1) << (num_bits - 1 - i)
            self._bit_reversal_cache[cache_key] = bit_reversed_indices
        return self._bit_reversal_cache[cache_key]
    
    def _get_twiddle_factors(self, L, stride):
        cache_key = (L, stride, str(self.device))
        if cache_key not in self._twiddle_cache:
            k_values = torch.arange(stride, device=self.device)
            angles = -2.0 * math.pi * k_values / L
            twiddles = torch.exp(1j * angles).to(torch.complex64)
            self._twiddle_cache[cache_key] = (twiddles.real, twiddles.imag, k_values)
        return self._twiddle_cache[cache_key]
    
    def _compute_loss(self, b, u, threshold=0.05, gate=1.0):
        if isinstance(b, tuple):
            br, bi = b; ur, ui = u
        else:
            br, bi, ur, ui = b.real, b.imag, u.real, u.imag
        def masked_mae(ref, est):
            diff = (est - ref).abs()                # (B, N)
            sel  = (ref.abs() > gate) & (diff > threshold)
            s = sel.to(diff.dtype)
            return (diff * s).sum(dim=1) / s.sum(dim=1).clamp_min(1)
        return masked_mae(br, ur), masked_mae(bi, ui)


    def _loss_scaling(self, binary, unary):
        '''
        binary, unary format: complex
        '''
        mae_loss_r, mae_loss_i = self._compute_loss(binary, unary)
        if self.verbose: print(f'MAE loss real: {mae_loss_r}, imag: {mae_loss_i}')
        allowed_loss = 0.2
        return torch.where((mae_loss_r > allowed_loss) | (mae_loss_i > allowed_loss), True, False)


    def binary_scale(self, input, quantile=0.95):
        assert 0.0 < quantile <= 1.0, "quantile must be in (0, 1]"
        x = input
        device = x.device
        rdtype = (x.real.dtype if torch.is_complex(x) else x.dtype)

        ql = 0.5 - quantile / 2.0
        qu = 0.5 + quantile / 2.0

        if torch.is_complex(x):
            # "shared logic" compute bounds over real+imag together
            comb = torch.cat([x.real.reshape(-1), x.imag.reshape(-1)])
            lower = torch.quantile(comb, ql)
            upper = torch.quantile(comb, qu)
        else:
            lower = torch.quantile(x, ql)
            upper = torch.quantile(x, qu)
        eps = 1e-12
        T = torch.max(lower.abs(), upper.abs())
        T = torch.clamp(T, min=eps)       
        k = -torch.ceil(torch.log2(T))     
        scale = torch.pow(torch.tensor(2.0, device=device, dtype=rdtype), k)
        output = x * scale
        return output, scale
    
    def update_butterfly_scale(self):
        with torch.no_grad():
            for adder in (
                self.butterfly_spike.add_y0r,
                self.butterfly_spike.add_y0i,
                self.butterfly_spike.add_y1r,
                self.butterfly_spike.add_y1i,
            ):
                adder.scale.copy_(adder.scale.new_tensor(self.scale))
    
    def forward(self, data, verbose=False):
        """
        Perform FFT computation on input data.
        
        Args:
            data (torch.Tensor): Input data of shape (B, N) where N must be power of 2
            
        Returns:
            tuple: (binary_output, unary_output) - FFT results
        """
        self.verbose = verbose  
        B, N = data.shape
        
        # Validate FFT size
        if (N & (N - 1)) != 0 or N == 0:
            raise ValueError("FFT size must be a power of 2.")
        
        if self.fft_size is not None and N != self.fft_size:
            #self.clear_cache() future work, in case we process different sizes
            raise ValueError(f"Input size {N} doesn't match expected FFT size {self.fft_size}")
        
        if data.device != self.device:
            data = data.to(self.device)
        if not torch.is_complex(data):
            data = data.to(torch.complex64)
        
        # bit-reversal
        bit_reversed_indices = self._get_bit_reversal_indices(N)
        binary_data = data[:, bit_reversed_indices].clone()

        # Perform FFT stages
        num_stages = int(math.log2(N))
        cum_binary_scale = torch.ones((), device=self.device, dtype=data.real.dtype)
        cum_unary_scale = torch.ones((), device=self.device, dtype=data.real.dtype)
        binary_data, stage_scale = self.binary_scale(binary_data, quantile=0.95)
        cum_binary_scale = cum_binary_scale * stage_scale
        unary_data = binary_data.clone()
        
        rp = 2  # Repeat flag, so it wont trigger first stage
        stage = 1
        self._current_timestep = self.codec_config['timestep']  
        while stage <= num_stages:
            target_timestep = 2 ** self.combo[stage - 1]
            self.codec_config['timestep'] = target_timestep
            self.mul_config['timestep'] = target_timestep
            self.butterfly_spike = butterfly_spike(
                self.codec_config, self.mul_config, self.add_config, self.acc_config
            ).to(self.device)
            width = int(math.log2(self.codec_config['timestep']))
            self.add_config['width'] = width + 1
            self._current_timestep = target_timestep
            
            L = 2 ** stage # butterfly size
            stride = L // 2 # the distance between pairs of butterflies
            wr_stage, wi_stage, k_values = self._get_twiddle_factors(L, stride)

            if rp == 0: #? RUN ONCE per batch
                unary_data, input_scale = self.binary_scale(unary_data, quantile=0.95) #! beacuse the input to each (stage =! 1) is unary not binary!!
                binary_data = binary_data * input_scale
                cum_binary_scale = cum_binary_scale * input_scale

            unary_data_copy = unary_data.clone()
            binary_data_copy = binary_data.clone() / self.scale  # scaled binary...
            for group_start in range(0, N, L):
                i0 = group_start + k_values
                i1 = i0 + stride

                # --- Binary path inputs
                x0b = binary_data_copy[:, i0]
                x1b = binary_data_copy[:, i1]
                x0b_r, x0b_i = x0b.real, x0b.imag
                x1b_r, x1b_i = x1b.real, x1b.imag

                # --- Unary path inputs
                x0u = unary_data[:, i0]
                x1u = unary_data[:, i1]
                x0u_r, x0u_i = x0u.real, x0u.imag
                x1u_r, x1u_i = x1u.real, x1u.imag

                # Twiddles (batched)
                wrb = wr_stage.unsqueeze(0).expand(B, -1)
                wib = wi_stage.unsqueeze(0).expand(B, -1)

                # --- Run both paths with their own inputs
                # unary
                y0u_r, y0u_i, y1u_r, y1u_i = self.butterfly_spike(
                    x0u_r, x0u_i, x1u_r, x1u_i, wrb, wib, timesteps=self.codec_config['timestep']
                )
                unary_data_copy[:, i0] = torch.complex(y0u_r, y0u_i)
                unary_data_copy[:, i1] = torch.complex(y1u_r, y1u_i)

                # binary (reference)
                y0b_r, y0b_i, y1b_r, y1b_i = self.butterfly_binary(
                    x0b_r, x0b_i, x1b_r, x1b_i, wrb, wib
                )
                binary_data_copy[:, i0] = torch.complex(y0b_r, y0b_i)
                binary_data_copy[:, i1] = torch.complex(y1b_r, y1b_i)

                self.butterfly_spike.reset() #! resets internal state of mul and add, timestep counter, spike counting, 
            
            loss = self._loss_scaling(binary_data_copy, unary_data_copy) 
             
            if self.verbose: print(f'Loss: {loss}, Scale: {self.scale}')
            repeat_mask = (((stage < 5) & (rp == 0 or rp == 2) & (self.scale < 2) & loss) |
                               ((stage == 5) & (self.scale < 5) & loss))
            
            repeat_percentage = repeat_mask.sum().float() / B
            if self.verbose: print(f'Repeat percentage: {repeat_percentage:.2f}')
            if repeat_percentage >= 0.05: 
                    self.scale += 1
                    self.add_config['scale'] = self.scale
                    self.update_butterfly_scale()
                    rp = 1  # Set repeat flag 
                    continue
            else:
                unary_data = unary_data_copy.clone()  # Update unary data for next stage
                binary_data = binary_data_copy.clone()  # Update binary data for next stage
                cum_unary_scale = cum_unary_scale * float(self.scale)
                self.scale = 1
                self.add_config['scale'] = self.scale
                self.update_butterfly_scale()
                rp = 0
            
            stage += 1  
        return binary_data, unary_data
    
    def clear_cache(self):
        """Clear cached bit-reversal indices and twiddle factors."""
        self._bit_reversal_cache.clear()
        self._twiddle_cache.clear()
