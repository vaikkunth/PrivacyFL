import random
import warnings
import datetime
import numpy as np

from initializer import Initializer

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    a = datetime.datetime.now()
    k = Initializer(num_clients=3, iterations=8, num_servers=1)
    k.run_simulation(8, server_agent_name='server_agent0')
    b = datetime.datetime.now()
    print('Time taken to run simulation script: {}'.format(b - a))
