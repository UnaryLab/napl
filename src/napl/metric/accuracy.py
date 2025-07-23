import torch

from napl.base import napl_base
from napl.utils import *
from loguru import logger


class accuracy(napl_base):
    """
    Calculate progressive accuracy based on progressive precision of input spike.
    Progressive precision: 'Fast and accurate computation using stochastic circuits'
    """
    def __init__(
            self, 
            config={
                'mode' : 'bipolar',
                'name' : 'accuracy_inst',
            }
        ):
        super().__init__()

        # check config
        check_config(config, ['mode', 'name'])

        # data representation
        self.mode = config['mode'].lower()
        check_mode(self.mode)
        
        self.name = config['name'].lower()

        self.timestep_cur = 0
        self.spike_count = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.spike_value = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.spike_error = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        # absolute max error
        self.spike_error_abs_max = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        # absolute min error
        self.spike_error_abs_min = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        # mean absolute error
        self.spike_error_mae = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        # root mean square error
        self.spike_error_rmse = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

        self.error_flag = False


    def reset(self):
        """
        Reset the timestep and one count.
        """
        self.timestep_cur = 0
        self.spike_count.data = torch.zeros(1)
        self.spike_value.data = torch.zeros(1)
        self.spike_error.data = torch.zeros(1)
        self.spike_error_abs_max.data = torch.zeros(1)
        self.spike_error_abs_min.data = torch.zeros(1)
        self.spike_error_mae.data = torch.zeros(1)
        self.spike_error_rmse.data = torch.zeros(1)
        self.error_flag = False
    

    def forward(self, spike: torch.Tensor):
        self.error_flag = True
        self.timestep_cur += 1
        # accuracy uses torch.float format to avoid overflow
        self.spike_count.data = self.spike_count.add(spike.type(torch.float))
        self.spike_value.data = self.spike_count.div(self.timestep_cur)
        if self.mode == 'bipolar':
            self.spike_value.data = self.spike_value.mul(2).sub(1)
        return self.spike_value
    

    def report_error(self, reference: torch.Tensor):
        assert self.error_flag == True, logger.error(f'Error flag is not set. Please call forward() before report_error().')
        self.spike_error.data = self.spike_value.sub(reference)
        self.spike_error_abs_max.data = torch.max(self.spike_error.abs())
        self.spike_error_abs_min.data = torch.min(self.spike_error.abs())
        self.spike_error_mae.data = self.spike_error.abs().mean()
        self.spike_error_rmse.data = torch.sqrt(self.spike_error.abs().pow(2).mean())

        logger.info(f'Accuracy report for accuracy instance <{self.name}> over <{self.timestep_cur}> timesteps: ')
        logger.info(f'    Max absolute error:     <{self.spike_error_abs_max.item()}>')
        logger.info(f'    Min absolute error:     <{self.spike_error_abs_min.item()}>')
        logger.info(f'    Mean absolute error:    <{self.spike_error_mae.item()}>')
        logger.info(f'    Root mean square error: <{self.spike_error_rmse.item()}>')
        logger.info(f'')
        return self.spike_error
    
