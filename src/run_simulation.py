import random
import warnings
import datetime
import config
import numpy as np

from initializer import Initializer

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    initializer = Initializer(num_clients=config.NUM_CLIENTS, iterations=config.ITERATIONS,
                              num_servers=config.NUM_SERVERS)
    # can use any amount of iterations less than config.ITERATIONS but the
    #  initializer has only given each client config.ITERATIONS datasets for training.
    a = datetime.datetime.now()
    initializer.run_simulation(config.ITERATIONS,
                               server_agent_name='server_agent0')
    b = datetime.datetime.now()