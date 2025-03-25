import yaml
import os

def create_block(A, B_block, start, ncols):
    """ 
    Copies `ncols` of `A` into contiguous memory `B_block`, starting from column `start`.

    Params:
        - A: NDArray
            Matrix to get columns from.
        - B_block: NDArray
            Matrix to copy columns to.
        - start: int
            Index of the first column.
        - ncols: int
            Number of columns to copy.
    """
    B_block[:,:] = A[:,start:start+ncols]

def read_config(config_path: str):
    """
    Read configuration for an experiment from a yaml file.

    Params:
      - config_path : str
        Path to the config file, without the extension.
    Returns:
      - params:
        Dictionary with parameters.
    """
    path = os.path.abspath(f'{config_path}.yaml')
    with open(path,'r') as ff:
        params = yaml.safe_load(ff)
    return params

def custom_warning(message, category, filename, lineno, line = None):
    return f"{filename}:\n{category.__name__}: {message}\n"
