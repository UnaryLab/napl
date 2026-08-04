import torch

class butterfly_binary(torch.nn.Module):
    r"""
    Evaluate an exact radix-2 complex butterfly in the binary domain.

    Use this stateless PyTorch module as a reference for
    :class:`butterfly_spike` or wherever spike-stream simulation is unnecessary.

    The precise target is the radix-2 decimation-in-time butterfly on complex
    inputs :math:`x_0`, :math:`x_1` and twiddle factor :math:`w`,

    .. math::

       y_0 = x_0 + w x_1,\qquad y_1 = x_0 - w x_1.

    The module evaluates that target exactly on the real and imaginary parts,

    .. math::

       t_r = w_r x_{1r} - w_i x_{1i},\qquad
       t_i = w_r x_{1i} + w_i x_{1r},

    .. math::

       y_{0r} = x_{0r} + t_r,\quad y_{0i} = x_{0i} + t_i,\quad
       y_{1r} = x_{0r} - t_r,\quad y_{1i} = x_{0i} - t_i,

    so the only error is floating-point rounding.

    .. rubric:: Example

    .. code-block:: python

        import torch
        from napl.sim.algorithm.fft import butterfly_binary

        operation = butterfly_binary()
        inputs = tuple(torch.zeros(1) for _ in range(6))
        y0r, y0i, y1r, y1i = operation(*inputs)
    """


    def __init__(self):
        """
        Construct the stateless binary-domain butterfly.

        .. container:: api-parameter-list

            **Parameters:**

            ``None``.
        """
        super().__init__()


    def forward(self, x0r, x0i, x1r, x1i, wr, wi):
        """
        Compute one exact complex butterfly.

        Args:
            x0r: Real part of the first complex input.
            x0i: Imaginary part of the first complex input.
            x1r: Real part of the second complex input.
            x1i: Imaginary part of the second complex input.
            wr: Real part of the twiddle factor.
            wi: Imaginary part of the twiddle factor.

        Returns:
            A tuple ``(y0r, y0i, y1r, y1i)`` of tensors with the broadcast
            input shape.

        This method creates no persistent state and does not modify its inputs.

        **Example:**

        .. code-block:: python

            outputs = operation(*inputs)
        """
        t_r = wr * x1r - wi * x1i
        t_i = wr * x1i + wi * x1r

        y0r = x0r + t_r
        y0i = x0i + t_i
        y1r = x0r - t_r
        y1i = x0i - t_i

        return y0r, y0i, y1r, y1i
