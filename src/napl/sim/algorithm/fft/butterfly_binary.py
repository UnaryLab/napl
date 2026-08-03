import torch

class butterfly_binary(torch.nn.Module):
    """
    Evaluate an exact radix-2 complex butterfly in the binary domain.

    Use this stateless PyTorch module as a reference for
    :class:`butterfly_spike` or wherever spike-stream simulation is unnecessary.

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
