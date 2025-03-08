import yaml
import os

def create_block(A, B_block, start, ncols):
    B_block[:,:] = A[:,start:start+ncols]

def read_config(config_path: str):
    """
    Read configuration for an experiment from a yaml file.

    Params:
      - config_path : str
        Path to the config file, withot the extension.
    Returns:
      - params:
        Dictionary with parameters.
    """
    path = os.path.abspath(f'{config_path}.yaml')
    with open(path,'r') as ff:
        params = yaml.safe_load(ff)
    return params
