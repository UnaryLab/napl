import torch

from napl.sim.base import napl_base
from napl.utils import pow2_lshift, pow2_rshift

class _round_ste_fn(torch.autograd.Function):
    """
    Straight-through rounding: round to a fixed-point grid on the forward pass, pass
    the gradient through unchanged on the backward pass. Plain round()/floor()/ceil()
    have zero gradient everywhere, which is useless for quantization-aware training.
    Semantics: round(x << f).clamp(min,max) >> f, using the float-safe pow2 shift shims
    instead of integer operators.
    """
    @staticmethod
    def forward(ctx, input, fracwidth, min_val, max_val):
        # pow2_lshift returns a fresh tensor, so in-place rounding cannot modify input.
        scaled = pow2_lshift(input, fracwidth)
        scaled.round_().clamp_(min_val, max_val)
        return pow2_rshift(scaled, fracwidth)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None

def round_ste(input, fracwidth=0, min_val=None, max_val=None):
    """
    Round a tensor to a fixed-point grid with a straight-through gradient.

    Args:
        input: Floating-point tensor to quantize.
        fracwidth: Number of fractional bits; the default is ``0``.
        min_val: Optional minimum scaled integer code. ``None`` disables the lower clamp.
        max_val: Optional maximum scaled integer code. ``None`` disables the upper clamp.

    Returns:
        Quantized tensor with the same dtype and shape as ``input``. During
        backpropagation, its input gradient is passed through unchanged.

    **Example:**

    .. code-block:: python

        import torch
        from napl.sim.operation.round_fxp import round_ste

        output = round_ste(torch.tensor([0.3]), fracwidth=2)
    """
    if min_val is None:
        min_val = float('-inf')
    if max_val is None:
        max_val = float('inf')
    input_float = input if input.dtype == torch.float32 else input.to(torch.float32)
    output = _round_ste_fn.apply(input_float, fracwidth, min_val, max_val)
    return output if input.dtype == torch.float32 else output.to(input.dtype)

class round_fxp(napl_base):
    """
    Quantize a tensor to a signed fixed-point format.

    This single-shot binary-domain kernel rounds to increments of
    ``2**(-fracwidth)`` and clamps to the representable range. Use it for
    quantization-aware training because the input gradient passes through unchanged.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl import round_fxp

        operation = round_fxp({'intwidth': 3, 'fracwidth': 4})
        output = operation(torch.tensor([0.1, -0.3]))
    """
    #: Marks this quantizer as a single-shot tensor operation.
    streaming = False
    def __init__(
            self,
            config={
                'intwidth': 3,
                'fracwidth': 4,
            }
        ):
        """
        Configure the signed fixed-point format.

        .. container:: api-parameter-list

            **Parameters:**

            - **config** – Configuration mapping.

              - **intwidth**: Number of integer magnitude bits; the default is ``3``.
              - **fracwidth**: Number of fractional bits; the default is ``4``.
              - **name**: Optional module name.
        """
        super().__init__(config, ['intwidth', 'fracwidth'])

        #: Number of integer magnitude bits in the signed fixed-point format.
        self.intwidth = config['intwidth']
        #: Number of fractional bits in the signed fixed-point format.
        self.fracwidth = config['fracwidth']
        #: Largest scaled integer retained before conversion back to a tensor value.
        self.max_val = 2**(self.intwidth + self.fracwidth) - 1
        #: Smallest scaled integer retained before conversion back to a tensor value.
        self.min_val = 1 - 2**(self.intwidth + self.fracwidth)
        # The RTL saturating clamp is combinational.
        #: Modeled scalar latency of the single-shot quantizer.
        self.delay = 0

    def _reset(self):
        """
        Reset no local state; this single-shot quantizer is stateless.
        """
        pass

    def forward(self, input):
        """
        Quantize a complete tensor with straight-through rounding.

        This stateless call does not advance a streaming timestep. Passing
        ``None`` returns ``None``.

        Args:
            input: Floating-point tensor to quantize, or ``None``.

        Returns:
            Quantized tensor with the same shape and dtype as ``input``, or
            ``None`` when ``input`` is ``None``.

        **Example:**

        .. code-block:: python

            output = operation(torch.tensor([0.1, -0.3]))
        """

        if input is None:
            return None
        return round_ste(input, self.fracwidth, self.min_val, self.max_val)
