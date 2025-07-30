import torch

from napl.base import global_config
from napl.utils import *
from napl.operation import sync_skewed


def test_sync_skewed():
    """
    Test sync_skewed with a simple configuration.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    sync_skewed_config={
        'width': 3
    }
    
    sync_skewed_inst = sync_skewed(sync_skewed_config).to(device)

    a = torch.tensor([[0, 0]]).type(global_config.stype).to(device)
    b = torch.tensor([[1, 1]]).type(global_config.stype).to(device)
    print(sync_skewed_inst(a,b))
    print(sync_skewed_inst.cnt)
    print()

    a = torch.tensor([[1, 1]]).type(global_config.stype).to(device)
    b = torch.tensor([[0, 0]]).type(global_config.stype).to(device)
    print(sync_skewed_inst(a,b))
    print(sync_skewed_inst.cnt)
    print()

    a = torch.tensor([[1, 1]]).type(global_config.stype).to(device)
    b = torch.tensor([[1, 1]]).type(global_config.stype).to(device)
    print(sync_skewed_inst(a,b))
    print(sync_skewed_inst.cnt)
    print()

    a = torch.tensor([[0, 0]]).type(global_config.stype).to(device)
    b = torch.tensor([[0, 0]]).type(global_config.stype).to(device)
    print(sync_skewed_inst(a,b))
    print(sync_skewed_inst.cnt)
    print()

    a = torch.tensor([[0, 0]]).type(global_config.stype).to(device)
    b = torch.tensor([[1, 1]]).type(global_config.stype).to(device)
    print(sync_skewed_inst(a,b))
    print(sync_skewed_inst.cnt)
    print()

    sync_skewed_inst.reset()

    print('Test passed.')


if __name__ == '__main__':
    test_sync_skewed()

