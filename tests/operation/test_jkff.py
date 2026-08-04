import torch

from napl.sim.base import global_config
from napl.sim.operation import jkff
from napl.utils._shared_test import devices, timer

    
def test_jkff():
    """Verify JK flip-flop state transitions and reset behavior for known J/K sequences."""

    first_j = torch.tensor([[0., 0., 1., 1.]]).type(global_config.stype)
    first_k = torch.tensor([[0., 1., 0., 1.]]).type(global_config.stype)
    second_j = torch.tensor([[1., 1., 0., 0.]]).type(global_config.stype)
    second_k = torch.tensor([[1., 0., 1., 0.]]).type(global_config.stype)

    for device in devices():
        jkff_inst = jkff().to(device)
        j = first_j.to(device)
        k = first_k.to(device)

        with timer(device) as elapsed:
            first_result = jkff_inst(j, k).detach().cpu().clone()
            second_result = jkff_inst(
                second_j.to(device),
                second_k.to(device),
            ).detach().cpu().clone()

        assert torch.equal(first_result, torch.tensor([[0, 0, 1, 1]], dtype=global_config.stype))
        assert torch.equal(second_result, torch.tensor([[1, 1, 0, 1]], dtype=global_config.stype))
        assert jkff_inst.timestep_cur == 2
        jkff_inst.reset()
        assert jkff_inst.timestep_cur == 0
        print(f'[{device}] time={elapsed.seconds * 1000:.3f}ms')

    print('Test passed.')


if __name__ == '__main__':
    test_jkff()
