import math
import subprocess
import sys
from pathlib import Path

import torch

from napl.sim.module import mgu_hard_mix
from napl.sim.operation import (add_gaines, add_ugemm, clamp_comp, clamp_sat, decorr,
                                div_cordiv, exp_n1, log_n1, negate, sub_scale, subabs,
                                tanh_p1)
from napl.syn import TranslationError, translate_node

#: Operation tree holding the RTL, its generators, and its generated vectors.
IMP = Path(__file__).resolve().parents[2] / "src" / "napl" / "imp" / "operation"

# The ROM contents are the hardware side of the add_gaines and decorr checks, and
# they are read from the generated artifact rather than recomputed here, so the
# clause's own sequence comparison is not what decides the check. div_cordiv has
# no ROM: its index order is combinational, so `rtl_index` below transcribes the
# gate map in div_cordiv.v.
#
# The grid covers dimensions past 64 because a clause that accepts only a listed
# range of dimensions rejects hardware-honored configurations there. Unscaled
# add_gaines rows are left out: their hardware form follows from add_gaines.v's
# unscaled branch being a plain OR with no ROM, which is a reading of the RTL
# rather than a measurement this sweep takes.
# The band above 64 stops at 96 to keep the sweep near ten seconds; it holds
# honored dimensions for every swept operation and depth (add_gaines 65, 75, 82,
# 90; div_cordiv 90 at depth 16) alongside dimensions the hardware disagrees with.
ADD_GAINES_DIMS = range(1, 97)
DIV_CORDIV_DEPTHS = (2, 4, 8, 16)
DIV_CORDIV_DIMS = range(1, 65)
DIV_CORDIV_DIMS_ABOVE_64 = range(65, 97)
DECORR_DEPTHS = (1, 2, 3, 4, 5, 8)
DECORR_TIMESTEPS = (64, 128, 256, 512)
# Entry 2 over 1 or 5 steps puts twice the bipolar gap peak at 2**width - 1 for
# width 2 or 3, the rows where the bipolar bound's + 2 decides acceptance.
ADD_UGEMM_ENTRIES = (1, 2, 3, 8, 9)
ADD_UGEMM_SEGMENTS = (1, 2, 5, 16)
ADD_UGEMM_WIDTHS = range(2, 11)


def rom_rows(operation):
    """Rows of an operation's generated position ROM.

    The vectors are build artifacts rather than checked-in files, so an absent
    ROM is regenerated from the operation's own generator; a generator failure
    fails the test. A generator rewrites all three files in its unit's vec
    directory, six across add_gaines and decorr, so a simulation reading them in
    another process can see one truncated mid-rewrite even though the bytes are
    deterministic.
    """
    path = IMP / operation / "vec" / f"{operation}_rom.hex"
    if not path.exists():
        subprocess.run([sys.executable, str(IMP / operation / "gen" / f"gen_{operation}.py")],
                       cwd=IMP / operation, check=True)
    return path.read_text(encoding="utf-8").split()


def accepts(node):
    """Whether translate resolves the node into an RTL binding."""
    try:
        translate_node(node)
        return True
    except TranslationError:
        return False


def rtl_index(count, width):
    """One index div_cordiv.v's `g_sobol` gates drive from counter value `count`.

    Bit 0 is the counter's top bit and bit ``width - 1 - b`` is the
    exclusive-or of counter bits ``b`` and ``b + 1``.
    """
    bit = lambda value, position: (value >> position) & 1
    bits = [0] * width
    bits[0] = bit(count, width - 1)
    for position in range(width - 1):
        bits[width - 1 - position] = bit(count, position) ^ bit(count, position + 1)
    return sum(value << index for index, value in enumerate(bits))


def add_gaines_rows():
    """(label, hardware honors it, translate accepts it) per add_gaines config."""
    rom = [int(row, 2) for row in rom_rows("add_gaines")]
    rows = []
    for entry in (2, 8, 16):
        for dim in (ADD_GAINES_DIMS if entry == 8 else range(1, 9)):
            config = {'polarity': 'bipolar', 'scaled': True, 'entry': entry,
                      'generator': 'sobol', 'dim': dim}
            # The ROM is the selector list the hardware walks, so a model whose
            # selectors are those rows in that order is the hardware's own.
            honored = list(add_gaines(config).sel_seq) == rom
            rows.append((f'entry={entry} dim={dim}', honored,
                         accepts({'class': 'add_gaines', 'config': config})))
    return rows


def div_cordiv_rows():
    """(label, hardware honors it, translate accepts it) per div_cordiv config."""
    rows = []
    for depth in DIV_CORDIV_DEPTHS:
        hardware = [rtl_index(count, int(math.log2(depth))) for count in range(depth)]
        dims = list(DIV_CORDIV_DIMS)
        if depth > 2:
            dims += list(DIV_CORDIV_DIMS_ABOVE_64)
        for dim in dims:
            config = {'depth': depth, 'generator': 'sobol', 'dim': dim}
            honored = div_cordiv(config).rand_seq_idx == hardware
            rows.append((f'depth={depth} dim={dim}', honored,
                         accepts({'class': 'div_cordiv', 'config': config})))
    return rows


def decorr_rows():
    """(label, hardware honors it, translate accepts it) per decorr config."""
    rom = rom_rows("decorr")
    rows = []
    for depth in DECORR_DEPTHS:
        index_width = 1 if depth <= 2 else int(math.ceil(math.log2(depth)))
        for timestep in DECORR_TIMESTEPS:
            config = {'polarity': 'unipolar', 'depth': depth, 'timestep': timestep,
                      'generator': 'sys', 'seed': 7}
            first, second = decorr(config).rand_seq_idx
            words = [f'{one:0{index_width}b}{zero:0{index_width}b}'
                     for zero, one in zip(first, second)]
            rows.append((f'depth={depth} timestep={timestep}', words == rom,
                         accepts({'class': 'decorr', 'config': config})))
    return rows


def add_ugemm_trace(config, stimulus):
    """Accumulator states the simulation model walks through over one run."""
    model = add_ugemm(dict(config))
    states = []
    for row in stimulus:
        model(torch.tensor(row, dtype=torch.float32), dim=-1)
        states.append((model.accumulator.item(),
                       model.out_accumulator.item() if not config['scaled'] else None))
    return states


def add_ugemm_rows():
    """(label, hardware honors it, translate accepts it) per add_ugemm config.

    The RTL has no accumulator clamp, so a configuration is honored exactly when
    the model's clamp never engages: its states match a 40-bit model's on the
    runs that reach the peaks, all ones for the high peak, all zeros for the
    bipolar low one, and ENTRY-1 ones then all ones for the scaled pre-carry peak.
    """
    rows = []
    for polarity in ("unipolar", "bipolar"):
        for scaled in (True, False):
            for entry in ADD_UGEMM_ENTRIES:
                for segment in ADD_UGEMM_SEGMENTS:
                    runs = ([[1] * entry] * segment, [[0] * entry] * segment,
                            [[1] * (entry - 1) + [0]] + [[1] * entry] * (segment - 1))
                    wide = {'polarity': polarity, 'scaled': scaled, 'width': 40}
                    reference = [add_ugemm_trace(wide, run) for run in runs]
                    for width in ADD_UGEMM_WIDTHS:
                        config = dict(wide, width=width)
                        try:
                            honored = [add_ugemm_trace(config, run) for run in runs] == reference
                        except AssertionError:
                            honored = False
                        rows.append((f'add_ugemm {polarity} scaled={scaled} entry={entry} '
                                     f'segment={segment} width={width}', honored,
                                     accepts({'class': 'add_ugemm', 'config': config,
                                              'inputs': {'input': {'shape': (4, entry)}},
                                              'segment': segment})))
    return rows


def constructs(cls, *args, **kwargs):
    """Whether the simulation constructor accepts the arguments."""
    try:
        cls(*args, **kwargs)
        return True
    except AssertionError:
        return False


def polarity_rows():
    """(label, hardware honors it, translate accepts it) per single-polarity config.

    Each circuit implements the one polarity its simulation constructor accepts,
    so a configuration is honored exactly when that constructor builds it. The
    other settings stay at the verified operating point, so polarity alone
    decides each row.
    """
    rows = []
    for cls in (exp_n1, log_n1, tanh_p1):
        for polarity, timestep in (("unipolar", 256), ("unipolar", 200),
                                   ("bipolar", 256)):
            config = {'polarity': polarity, 'timestep': timestep, 'generator': 'sobol',
                      'dim': 1}
            rows.append((f'{cls.__name__} {polarity} timestep={timestep}',
                         constructs(cls, dict(config)),
                         accepts({'class': cls.__name__, 'config': config})))
    for polarity in ("unipolar", "bipolar"):
        for cls in (negate, subabs):
            config = {'polarity': polarity}
            rows.append((f'{cls.__name__} {polarity}', constructs(cls, dict(config)),
                         accepts({'class': cls.__name__, 'config': config})))
        config = {'polarity': polarity, 'scale': 2, 'intwidth': 8, 'fracwidth': 0}
        rows.append((f'sub_scale {polarity}', constructs(sub_scale, dict(config)),
                     accepts({'class': 'sub_scale', 'config': config})))
        weights = {'weight_f': torch.zeros(3, 7), 'bias_f': torch.zeros(3),
                   'weight_n': torch.zeros(3, 7), 'bias_n': torch.zeros(3),
                   'hx_value': torch.zeros(1, 3)}
        config = {'polarity': polarity, 'timestep': 256, 'generator': 'sobol',
                  'width': 10, 'depth_ismul': 6}
        rows.append((f'mgu_hard_mix {polarity}',
                     constructs(mgu_hard_mix, **weights, config=dict(config)),
                     accepts({'class': 'mgu_hard_mix',
                              'config': dict(weights, config=config, lanes=3)})))
    return rows


def boundary_rows():
    """(label, hardware honors it, translate accepts it) at constructor bounds.

    A configuration is honored exactly when its simulation constructor builds
    it. Each row sits one grid step inside or outside a bound the constructor
    checks, with every other setting at the verified operating point, so that
    bound alone decides the row.
    """
    rows = []
    for polarity, scaled, extra in (("unipolar", False, {'entry': 2}),
                                    ("bipolar", False, {'entry': 2}),
                                    ("unipolar", True, {'entry': 8, 'generator': 'sobol'}),
                                    ("bipolar", True, {'entry': 8, 'generator': 'sobol'}),
                                    ("bipolar", True, {'entry': 8})):
        config = dict({'polarity': polarity, 'scaled': scaled, 'dim': 5}, **extra)
        rows.append((f'add_gaines {config}', constructs(add_gaines, dict(config)),
                     accepts({'class': 'add_gaines', 'config': config})))
    # One step of the 2**-8 bound grid both clamp classes resolve.
    step = 2.0 ** -8
    for cls in (clamp_sat, clamp_comp):
        for polarity in ("unipolar", "bipolar"):
            low = -1.0 if polarity == "bipolar" else 0.0
            for lo, hi in ((low, 0.5), (low - step, 0.5), (0.0, 1.0), (0.0, 1.0 + step),
                           (0.5 - step, 0.5), (0.5, 0.5)):
                config = {'polarity': polarity, 'lo': lo, 'hi': hi}
                rows.append((f'{cls.__name__} {config}', constructs(cls, dict(config)),
                             accepts({'class': cls.__name__, 'config': config})))
    return rows


#: Configurations the RTL co-simulation covers: gen_encode.py's DIMS,
#: gen_div_cordiv.py's depth, and gen_exp_n2g.py's config.
ENCODE_DIMS = (1, 2, 3, 4)
ENCODE_NON_SOBOL = ('lfsr', 'lfsr_ext', 'tc', 'temporal')
DIV_CORDIV_DEPTH = 2
EXP_N2G_CONFIG = {'depth': 5, 'gain': 1}


def rtl_bound_rows():
    """(label, hardware honors it, translate accepts it) at RTL-only bounds.

    Each row sits at or one step past a bound the simulator does not share, so
    a configuration is honored exactly when the co-simulation covers it.
    """
    rows = []
    for dim in range(1, 6):
        config = {'polarity': 'unipolar', 'timestep': 256, 'generator': 'sobol', 'dim': dim}
        rows.append((f'encode dim={dim}', dim in ENCODE_DIMS,
                     accepts({'class': 'encode', 'config': config})))
    # The other generators draw the same sequence at every dimension and read no
    # direction-vector table, so gen_encode.py's dimension-1 run covers dim 5.
    for generator in ENCODE_NON_SOBOL:
        config = {'polarity': 'unipolar', 'timestep': 256, 'generator': generator, 'dim': 5}
        rows.append((f'encode {generator} dim=5', True,
                     accepts({'class': 'encode', 'config': config})))
    # rc and rate share the Sobol branch, so at dim 5 they draw a sequence the
    # RTL holds no table for.
    for generator in ('rc', 'rate'):
        config = {'polarity': 'unipolar', 'timestep': 256, 'generator': generator, 'dim': 5}
        rows.append((f'encode {generator} dim=5', False,
                     accepts({'class': 'encode', 'config': config})))
    for depth in (1, 2):
        config = {'depth': depth, 'generator': 'sobol'}
        rows.append((f'div_cordiv depth={depth}', depth == DIV_CORDIV_DEPTH,
                     accepts({'class': 'div_cordiv', 'config': config})))
    # gen_exp_n2g.py's gain is the simulator's default, so leaving it out
    # names the same configuration.
    for config in (dict(EXP_N2G_CONFIG), {'depth': EXP_N2G_CONFIG['depth']}):
        rows.append((f'exp_n2g {config}', True,
                     accepts({'class': 'exp_n2g', 'config': config})))
    return rows


def test_requires_clauses_accept_every_configuration_the_hardware_honors():
    """Translate accepts a configuration exactly when the generated hardware honors it."""
    for name, rows in (("add_gaines", add_gaines_rows()),
                       ("div_cordiv", div_cordiv_rows()),
                       ("decorr", decorr_rows()),
                       ("add_ugemm", add_ugemm_rows()),
                       ("polarity", polarity_rows()),
                       ("boundary", boundary_rows()),
                       ("rtl_bound", rtl_bound_rows())):
        false_rejects = [label for label, honored, accepted in rows if honored and not accepted]
        false_accepts = [label for label, honored, accepted in rows if accepted and not honored]
        assert not false_rejects, f"{name} rejects configurations the hardware honors: {false_rejects}"
        assert not false_accepts, f"{name} accepts configurations the hardware cannot honor: {false_accepts}"
        assert any(honored for _, honored, _ in rows), f"{name} grid holds no honored configuration"
        assert any(not honored for _, honored, _ in rows), f"{name} grid holds no rejected configuration"

    # A dimension past 64 that the hardware honors is the case an enumerated
    # dimension list cannot reach.
    assert accepts({'class': 'add_gaines',
                    'config': {'polarity': 'bipolar', 'scaled': True, 'entry': 8,
                               'generator': 'sobol', 'dim': 90}})
    assert accepts({'class': 'div_cordiv',
                    'config': {'depth': 16, 'generator': 'sobol', 'dim': 90}})


if __name__ == "__main__":
    test_requires_clauses_accept_every_configuration_the_hardware_honors()
    print("Test passed.")
