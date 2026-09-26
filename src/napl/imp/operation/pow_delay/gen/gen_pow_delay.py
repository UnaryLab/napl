"""
Generate golden vectors for the pure-delay ``pow_delay`` RTL from the Python
model. The generated N parameter is the single sizing source for the model,
both polarity variants, and the testbench elaboration.

Output rows are ``<rst> <input> <unipolar> <bipolar> <deep-unipolar>
<deep-bipolar>``. The deep pair uses depth 3 so the multi-register parameter
path is co-simulated. A set ``rst`` bit resets the Python models before that
input and pulses the RTL active-low reset before the corresponding vector is
checked.
"""
import sys
from pathlib import Path

import torch
from napl.sim.operation import pow_delay
from napl.syn import translate_node

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _gen_common import encode_value, rep_values


VEC = Path(__file__).resolve().parent.parent / 'vec' / 'pow_delay.vec'
PARAMS = Path(__file__).resolve().parent.parent / 'vec' / 'pow_delay_params.vh'

N = 3
DEPTH = 1
DEEP_DEPTH = 3
TIMESTEP = 256
CONFIG = {'polarity': 'unipolar', 'n': N, 'depth': DEPTH}
CODEC = {'polarity': 'bipolar', 'timestep': TIMESTEP,
         'generator': 'sobol', 'dim': 1}

_VALUES = rep_values(CODEC['polarity'])
_MID = len(_VALUES) // 2

DRIVE = []
for _i, _v in enumerate(_VALUES):
    _segment = encode_value(CODEC, _v)
    for _j, _spike in enumerate(_segment):
        DRIVE.append((1 if (_i == _MID and _j == 0) else 0, _spike))

_REPLAY = encode_value(CODEC, 0.375)[:64]
_DIRTY = encode_value(CODEC, 0.875)[:64]
_REPLAY_START = len(DRIVE)
DRIVE.extend((1 if index == 0 else 0, spike)
             for index, spike in enumerate(_REPLAY))
DRIVE.extend((0, spike) for spike in _DIRTY)
_REPLAY_AGAIN_START = len(DRIVE)
DRIVE.extend((1 if index == 0 else 0, spike)
             for index, spike in enumerate(_REPLAY))


def run(polarity, depth):
    """Return the Python trace for one polarity."""
    model = pow_delay(config=dict(CONFIG, polarity=polarity, depth=depth))
    model.reset()
    outputs = []
    for reset, spike in DRIVE:
        if reset:
            model.reset()
        value = torch.tensor(spike, dtype=model.stype)
        outputs.append(int(model(value).item()))
    return outputs


def main():
    """Generate vectors and require the mapping to carry the same N."""
    VEC.parent.mkdir(parents=True, exist_ok=True)
    model = pow_delay(config=CONFIG)
    out_uni = run('unipolar', DEPTH)
    out_bi = run('bipolar', DEPTH)
    out_deep_uni = run('unipolar', DEEP_DEPTH)
    out_deep_bi = run('bipolar', DEEP_DEPTH)

    replay_stop = _REPLAY_START + len(_REPLAY)
    replay_again_stop = _REPLAY_AGAIN_START + len(_REPLAY)
    assert out_uni[_REPLAY_START:replay_stop] == out_uni[_REPLAY_AGAIN_START:replay_again_stop]
    assert out_bi[_REPLAY_START:replay_stop] == out_bi[_REPLAY_AGAIN_START:replay_again_stop]
    assert out_deep_uni[_REPLAY_START:replay_stop] == out_deep_uni[_REPLAY_AGAIN_START:replay_again_stop]
    assert out_deep_bi[_REPLAY_START:replay_stop] == out_deep_bi[_REPLAY_AGAIN_START:replay_again_stop]

    for polarity in ('unipolar', 'bipolar'):
        for depth in (DEPTH, DEEP_DEPTH):
            binding = translate_node({
                'class': 'pow_delay',
                'config': dict(CONFIG, polarity=polarity, depth=depth),
            })
            expected = {'N': N, 'DEPTH': depth}
            assert binding.parameters == expected, (
                f'mapping pow_delay_{polarity} resolves {binding.parameters}, '
                f'not {expected}'
            )

    PARAMS.write_text(
        f'`define GEN_N {N}\n'
        f'`define GEN_DEPTH {DEPTH}\n'
        f'`define GEN_DEEP_DEPTH {DEEP_DEPTH}\n'
        f'`define GEN_PP_DELAY {model.hw.pp_delay}\n'
        f'`define GEN_VECTORS {len(DRIVE)}\n'
    )
    with VEC.open('w') as output:
        for row in zip(DRIVE, out_uni, out_bi, out_deep_uni, out_deep_bi):
            (reset, spike), uni, bi, deep_uni, deep_bi = row
            output.write(
                f'{reset} {spike} {uni} {bi} {deep_uni} {deep_bi}\n'
            )

    print(
        f'wrote {VEC} ({len(DRIVE)} vectors) and {PARAMS} '
        f'(GEN_N={N}, GEN_DEPTH={DEPTH}, GEN_DEEP_DEPTH={DEEP_DEPTH}, '
        f'GEN_PP_DELAY={model.hw.pp_delay})'
    )


if __name__ == '__main__':
    main()
