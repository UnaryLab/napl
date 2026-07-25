from typing import NamedTuple

import torch
from loguru import logger


class Analysis(NamedTuple):
    absolute: torch.Tensor
    absolute_min: torch.Tensor
    absolute_max: torch.Tensor
    mean: torch.Tensor
    mean_absolute: torch.Tensor
    root_mean_square: torch.Tensor
    max_absolute_index: torch.Tensor


def analyze(
        input: torch.Tensor,
        *,
        verbose=False,
        report=None,
        value=None,
        timestep=None,
    ) -> Analysis:
    absolute = input.abs()
    absolute_min, absolute_max = torch.aminmax(absolute)
    result = Analysis(
        absolute=absolute,
        absolute_min=absolute_min,
        absolute_max=absolute_max,
        mean=input.mean(),
        mean_absolute=absolute.mean(),
        root_mean_square=torch.sqrt(absolute.pow(2).mean()),
        max_absolute_index=torch.argmax(absolute),
    )
    if verbose:
        if timestep is None:
            logger.info(f'{report} report: ')
        else:
            logger.info(f'{report} report over <{timestep}> timesteps: ')
        logger.info(f'    Max absolute {value}:     <{result.absolute_max.item()}>')
        logger.info(f'    Min absolute {value}:     <{result.absolute_min.item()}>')
        logger.info(f'    Mean {value}:             <{result.mean.item()}>')
        logger.info(f'    Mean absolute {value}:    <{result.mean_absolute.item()}>')
        logger.info(f'    Root mean square {value}: <{result.root_mean_square.item()}>')
        logger.info('')
    return result
