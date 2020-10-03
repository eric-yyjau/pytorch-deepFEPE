
import logging

# from model_utils import set_nan2zero
def set_nan2zero(tens, name = 'network'):
    """
    # set nans in tensor to zeors. still need to check where the nans are!!!
    # for tensors
    """
    mat_nans = (tens != tens)
    n_nans = mat_nans.sum()
    if n_nans > 0:
        logging.warning(f"{name} include {n_nans} nans!!")
    tens[mat_nans] = 0 
    return tens