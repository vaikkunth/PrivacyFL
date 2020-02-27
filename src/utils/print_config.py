import sys

sys.path.append('..')
import config

def print_config(len_per_iteration):
    """
    Prints parameters at start of simulation. The two arguments are dynamically created and hence not in config.
    :param len_per_iteration: length of training dataset for each client for each iteration
    :param sensitivity: sensitivity for differential privacy
    """
    print('\n')
    print(
        'Running simulation with: \n{} clients \n{} iterations \n{}differential privacy \nand {}security \n'.format(
            config.NUM_CLIENTS, config.ITERATIONS, 'no ' if not config.USE_DP_PRIVACY else '',
            'no ' if not config.USE_SECURITY else ''))
    print('Training length per client per iteration is {}\n'.format((len_per_iteration)))
    print(
        'Simulation parameters are: \nTolerance for weight convergence = {} \nEpsilon for DP privacy is {}'.format(
            config.tolerance, config.epsilon))
