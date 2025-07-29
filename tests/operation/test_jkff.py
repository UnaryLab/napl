import torch, math

from napl.base import global_config
from napl.utils import *
from napl.operation import jkff

    
def test_jkff():
    """
    Test jkff with a simple configuration.
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    jkff_inst = jkff().to(device)

    j = torch.tensor([[0., 0., 1., 1.]]).type(global_config.ntype).to(device)
    k = torch.tensor([[0., 1., 0., 1.]]).type(global_config.ntype).to(device)

    print(jkff_inst(j,k))

    j = torch.tensor([[1., 1., 0., 0.]]).type(global_config.ntype).to(device)
    k = torch.tensor([[1., 0., 1., 0.]]).type(global_config.ntype).to(device)

    print(jkff_inst(j,k))

    print('Test passed.')


if __name__ == '__main__':
    test_jkff()

