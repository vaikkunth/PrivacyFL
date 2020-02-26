import sys
sys.path.append('..')

import random
import warnings
import datetime
import config
import numpy as np

from initializer_serverless import InitializerServerless


if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    initializer = InitializerServerless(num_clients=config.NUM_CLIENTS, iterations=config.ITERATIONS)
    # can use any amount of iterations less than config.ITERATIONS but the
    #  initializer has only given each client config.ITERATIONS datasets for training.
    a = datetime.datetime.now()
    initializer.run_simulation(config.ITERATIONS)
    b = datetime.datetime.now()
