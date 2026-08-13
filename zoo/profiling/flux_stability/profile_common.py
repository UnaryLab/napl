"""Shared runner for operation and module flux-stability profiles.

Flux stability of one streaming op run is the mean over encoded input streams i of
    output_stability_scalar / input_stability_scalar_i,
where each input_stability_scalar_i is the mean over tensor dims of that input
stream's settled per-element stability, and output_stability_scalar is the mean
over output streams of each output stream's mean per-element stability
(mean-then-divide, averaging multiple outputs into one number before dividing).
"""

import torch
import torch.nn.functional as F

from napl.sim.operation import encode
from napl.sim.metric import accuracy, stability

# Denominator floor so a fully unstable input stream does not divide by zero.
_EPS = 1e-12


def _output_rmse(monitors, refs):
    """Return the mean per-stream RMSE from final accuracy errors."""
    errors = [monitor.analyze(ref)[0] for monitor, ref in zip(monitors, refs)]
    return torch.stack([error.square().mean().sqrt() for error in errors]).mean().item()


def profile_op(op_class, *, ctor, inputs, reference, apply=None,
               output_polarity=None, timesteps=256, seed=0, make_op=None):
    """Profile one streaming op's flux stability on perf-scale random inputs.

    Args:
        op_class: The streaming operation class to construct as ``op_class(ctor)``;
            may be ``None`` when ``make_op`` is given.
        make_op: Optional zero-arg callable returning the constructed op; when given
            it replaces ``op_class(ctor)`` (for ops whose ctor is multi-arg).
        ctor: Config dict passed to ``op_class``; its ``'polarity'`` sets the default
            output-monitor polarity.
        inputs: List of input specs. An encoded stream spec is
            ``{'range': (lo, hi), 'polarity': str, 'shape': tuple, 'dim': int,
            'generator': 'sobol', 'encode': True}`` (``dim`` defaults to its 1-based
            index, ``generator`` to ``'sobol'``, ``encode`` to ``True``); a raw-value
            operand spec is ``{'encode': False, 'range': (lo, hi), 'shape': tuple}``.
            An encoded input may add ``'reduce_dim': int``, meaning the op reduces
            that input over this dim, so each slice along it is one effective input.
        reference: Callable ``(values_tuple, polarity) -> tensor | tuple`` giving the
            binary-domain analytic output(s) used as the output-monitor source(s).
        apply: Callable ``(op, spikes_list, values_list) -> spike | tuple``; defaults
            to ``op(*spikes)`` over the encoded spikes.
        output_polarity: Polarity for the output-monitor source(s); defaults to
            ``ctor['polarity']``.
        timesteps: Number of streamed timesteps.
        seed: Torch manual seed for reproducibility.

    Returns:
        A result dict with keys ``flux_stability``, ``rmse``, ``polarity``, ``timesteps``,
        ``shape``, ``seed``, ``n_inputs`` (effective inputs, counting each reduced
        slice separately), and ``n_outputs``.
    """
    torch.manual_seed(seed)
    op = make_op() if make_op is not None else op_class(ctor)
    polarity = ctor['polarity']
    out_polarity = output_polarity if output_polarity is not None else polarity
    if apply is None:
        apply = lambda op, spikes, values: op(*spikes)

    # Draw one perf-scale value tensor per operand: a custom sampler if given, else uniform over its range.
    values = []
    for spec in inputs:
        sampler = spec.get('sample')
        if sampler is not None:
            values.append(sampler(spec['shape']))
        else:
            lo, hi = spec['range']
            values.append(lo + (hi - lo) * torch.rand(spec['shape']))

    # Build a Sobol encoder and stability monitor for each encoded input stream, mirroring _make_pipeline.
    encoders, in_monitors, encoded_index = [], [], []
    for i, (spec, value) in enumerate(zip(inputs, values)):
        if not spec.get('encode', True):
            encoders.append(None)
            continue
        codec = {
            'polarity': spec['polarity'],
            'timestep': timesteps,
            'generator': spec.get('generator', 'sobol'),
            'dim': spec.get('dim', i + 1),
        }
        encoders.append(encode(codec))
        in_monitors.append(stability(value, {'polarity': spec['polarity'], 'threshold': 0.05}))
        encoded_index.append(i)

    # Build one output stability monitor per analytic output stream.
    refs = reference(tuple(values), polarity)
    if not isinstance(refs, tuple):
        refs = (refs,)
    out_monitors = [stability(r, {'polarity': out_polarity, 'threshold': 0.05}) for r in refs]
    out_accuracy = [accuracy({'polarity': out_polarity}) for _ in refs]

    # Stream: encode each input, feed its input monitor, apply the op, feed each output monitor.
    for _ in range(timesteps):
        spikes = []
        for i, (enc, value) in enumerate(zip(encoders, values)):
            if enc is None:
                continue
            spikes.append(enc(value))
        for spike, mon in zip(spikes, in_monitors):
            mon(spike)
        out_spikes = apply(op, spikes, values)
        if not isinstance(out_spikes, tuple):
            out_spikes = (out_spikes,)
        for spike, mon, acc in zip(out_spikes, out_monitors, out_accuracy):
            mon(spike)
            acc(spike)

    output_stab = torch.stack([m.stability.mean() for m in out_monitors]).mean()

    # One denominator per effective input: a reducing input splits into one per slice
    # along its reduce dim, every other input contributes its single stability mean.
    denoms = []
    for mon, idx in zip(in_monitors, encoded_index):
        reduce_dim = inputs[idx].get('reduce_dim')
        if reduce_dim is None:
            denoms.append(mon.stability.mean())
        else:
            sliced = mon.stability.movedim(reduce_dim, 0).reshape(mon.stability.shape[reduce_dim], -1)
            denoms.extend(sliced.mean(dim=1).unbind())

    flux = torch.stack([output_stab / d.clamp_min(_EPS) for d in denoms]).mean().item()
    rmse = _output_rmse(out_accuracy, refs)

    first_encoded_shape = list(values[encoded_index[0]].shape)
    return {
        'flux_stability': flux,
        'rmse': rmse,
        'polarity': polarity,
        'timesteps': timesteps,
        'shape': first_encoded_shape,
        'seed': seed,
        'n_inputs': len(denoms),
        'n_outputs': len(out_monitors),
    }


def profile_module(make_op, *, activation, reference, polarity,
                   feedback=None, timesteps=256, seed=0):
    """Profile one streaming module's flux stability on perf-scale random inputs.

    Unlike :func:`profile_op`, the module is built by ``make_op`` with its weight
    and bias baked in; those parameter streams are generated internally and are
    NOT monitored, so only the single activation stream is an effective input and
    the sole term in the denominator. For a recurrent module the previous output
    spike is fed back as ``hx``, but ``hx`` is likewise NOT monitored: the
    activation stream is the only monitored effective input.

    Args:
        make_op: Zero-arg callable returning the constructed module.
        activation: Spec dict for the single activation stream: ``'range'``,
            ``'polarity'``, ``'shape'``, ``'dim'`` (Sobol dim, default 1),
            ``'generator'`` (default ``'sobol'``), plus exactly one denominator
            mode, either ``'reduce_dim': int`` or
            ``'unfold': {'kernel_size', 'stride', 'padding', 'fold_channels'}``.
        reference: Callable ``(activation_value, polarity) -> tensor | tuple``
            giving the binary-domain analytic output(s) used as monitor source(s).
        polarity: ``'unipolar'`` or ``'bipolar'``; the default output-monitor
            polarity and the activation polarity unless the spec overrides it.
        feedback: ``None`` for feedforward modules; for a recurrent module a dict
            ``{'init', 'polarity', 'dim'}`` seeding the step-0 ``hx`` spike.
        timesteps: Number of streamed timesteps.
        seed: Torch manual seed for reproducibility.

    Returns:
        A result dict with keys ``flux_stability``, ``rmse``, ``polarity``, ``timesteps``,
        ``shape``, ``seed``, ``n_inputs`` (effective inputs), and ``n_outputs``.
    """
    torch.manual_seed(seed)
    op = make_op()
    act_polarity = activation.get('polarity', polarity)

    # Draw the activation value tensor uniform over its range and build its encoder + monitor.
    lo, hi = activation['range']
    act_value = lo + (hi - lo) * torch.rand(activation['shape'])
    act_encoder = encode({
        'polarity': act_polarity,
        'timestep': timesteps,
        'generator': activation.get('generator', 'sobol'),
        'dim': activation.get('dim', 1),
    })
    act_monitor = stability(act_value, {'polarity': act_polarity, 'threshold': 0.05})

    # One output stability monitor per analytic output stream.
    refs = reference(act_value, polarity)
    if not isinstance(refs, tuple):
        refs = (refs,)
    out_monitors = [stability(r, {'polarity': polarity, 'threshold': 0.05}) for r in refs]
    out_accuracy = [accuracy({'polarity': polarity}) for _ in refs]

    # Recurrent module: seed hx from init once, then feed the previous output spike back.
    hx_spike = None
    if feedback is not None:
        hx_encoder = encode({
            'polarity': feedback['polarity'],
            'timestep': timesteps,
            'generator': activation.get('generator', 'sobol'),
            'dim': feedback['dim'],
        })
        hx_spike = hx_encoder(feedback['init'])

    # Stream: encode the activation, feed its monitor, call the module, feed each output monitor.
    for _ in range(timesteps):
        act_spike = act_encoder(act_value)
        act_monitor(act_spike)
        if feedback is None:
            out = op(act_spike)
        else:
            out = op(act_spike, hx_spike)
            hx_spike = out
        out_spikes = out if isinstance(out, tuple) else (out,)
        for spike, mon, acc in zip(out_spikes, out_monitors, out_accuracy):
            mon(spike)
            acc(spike)

    output_stab = torch.stack([m.stability.mean() for m in out_monitors]).mean()

    # One denominator per effective input, derived from the activation's per-element stability.
    S = act_monitor.stability
    if 'reduce_dim' in activation:
        # Reduce over the contract axis: each slice along it is one effective input.
        d = activation['reduce_dim']
        sliced = S.movedim(d, 0).reshape(S.shape[d], -1)
        denoms = list(sliced.mean(dim=1).unbind())
    else:
        # Unfold NCHW into sliding taps: each tap is one effective input.
        u = activation['unfold']
        U = F.unfold(S, kernel_size=u['kernel_size'], stride=u['stride'], padding=u['padding'])
        if u['fold_channels']:
            denoms = list(U.mean(dim=(0, 2)).unbind())
        else:
            k2 = u['kernel_size'] * u['kernel_size']
            U = U.reshape(U.shape[0], -1, k2, U.shape[2])
            denoms = list(U.mean(dim=(0, 1, 3)).unbind())

    flux = torch.stack([output_stab / d.clamp_min(_EPS) for d in denoms]).mean().item()
    rmse = _output_rmse(out_accuracy, refs)

    return {
        'flux_stability': flux,
        'rmse': rmse,
        'polarity': polarity,
        'timesteps': timesteps,
        'shape': list(act_value.shape),
        'seed': seed,
        'n_inputs': len(denoms),
        'n_outputs': len(out_monitors),
    }


if __name__ == '__main__':
    # Smoke test: profile bipolar mul_gaines (two encoded inputs, single a*b output).
    from napl.sim.operation import mul_gaines

    result = profile_op(
        mul_gaines,
        ctor={'polarity': 'bipolar'},
        inputs=[
            {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (4096,)},
            {'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (4096,)},
        ],
        reference=lambda values, polarity: values[0] * values[1],
        timesteps=256,
        seed=0,
    )
    print(result)

    # Smoke test: profile bipolar linear_ugemm as a module (activation is the only monitored input).
    from napl.sim.module import linear_ugemm

    weight = -1.0 + 2.0 * torch.rand(4, 8)
    module_result = profile_module(
        lambda: linear_ugemm(weight, None, {'polarity': 'bipolar', 'timestep': 256,
                                            'generator': 'sobol', 'dim': 1,
                                            'scale': None, 'width': 12}),
        activation={'range': (-1.0, 1.0), 'polarity': 'bipolar', 'shape': (32, 8),
                    'dim': 1, 'generator': 'sobol', 'reduce_dim': -1},
        reference=lambda a, pol: (a @ weight.T) / 8,
        polarity='bipolar',
    )
    print(module_result)
