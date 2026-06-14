"""
Timing primitives for napl kernel speedup measurement (the napl-opt-sim skill).

Why this exists: GPU execution is asynchronous, so a naive perf_counter() around a
kernel call times the launch, not the compute. Every number here is taken with an
explicit device synchronize bracketing the timed region (the rule CLAUDE.md's
testing section and napl-validate-unarysim both enforce). Import these helpers from a
small per-kernel timing script instead of re-deriving the sync dance each time, so
the before/after numbers are measured identically.
"""
import time
import torch


def device_list():
    """Every device the kernel must be timed on, per CLAUDE.md: CPU plus whatever
    GPU is present. On Apple silicon the GPU is MPS, which the common
    'cuda if available else cpu' idiom silently skips, so we check both backends."""
    devs = ['cpu']
    if torch.cuda.is_available():
        devs.append('cuda')
    if torch.backends.mps.is_available():
        devs.append('mps')
    return devs


def sync(device):
    """Block until all queued work on `device` has finished. No-op on CPU. Call this
    before stopping any GPU clock, or the measurement is meaningless."""
    d = str(device)
    if d.startswith('cuda'):
        torch.cuda.synchronize()
    elif d.startswith('mps'):
        torch.mps.synchronize()


def time_ms(fn, device, warmup=3, iters=10):
    """Median wall-clock milliseconds of `fn` on `device`.

    `fn` is a no-arg thunk that runs ONE full workload (e.g. module.reset() followed
    by the timesteps loop). It must be self-contained and start from a clean state
    each call so the iterations are comparable. Warmup runs are discarded (they pay
    first-call allocation, autotuning, and the lazy MPS graph build); the timed
    region is synchronized on both ends so async GPU work is fully accounted for.
    Median over `iters` is reported rather than mean to shrug off scheduler jitter."""
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return samples[len(samples) // 2]
