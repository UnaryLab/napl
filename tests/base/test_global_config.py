import os, torch

from napl.base import global_config


def test_global_config():
    """
    Test the global configuration loading.
    """
    assert global_config.config_file is not None, "Global config file should be set."
    assert os.path.exists(global_config.config_file), f"Global config file {global_config.config_file} does not exist."
    
    assert global_config.stype in [torch.float, torch.bfloat16, torch.int8], \
        f"Invalid spike type {global_config.stype}; legal types are: [torch.float, torch.bfloat16, torch.int8]."
    
    assert global_config.ntype in [torch.float, torch.bfloat16], \
        f"Invalid non-spike type {global_config.ntype}; legal types are: [torch.float, torch.bfloat16]."

    print(f"Global config file: {global_config.config_file}")
    print(f"Global non-spike type: {global_config.ntype}")
    print(f"Global spike type: {global_config.stype}")
    print("Global configuration test passed.")


if __name__ == "__main__":
    test_global_config() 
