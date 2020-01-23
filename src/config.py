"""
Parameters to configure for simulation
"""
from datetime import timedelta

NUM_CLIENTS = 3
NUM_SERVERS = 1 # NOTE: More than one server will require additional coding to specify your simulation
ITERATIONS = 8
LEN_TEST = 600 # note total length is like 1797
########################
USE_SECURITY = True
USE_DP_PRIVACY = False
# Latency variables
client_agent1 = {'client_agent1': timedelta(seconds=1), 'client_agent2': timedelta(seconds=2), 'server_agent0': timedelta(seconds=3)}
client_agent2 = {'client_agent0': timedelta(seconds=1), 'client_agent2': timedelta(seconds=4), 'server_agent0': timedelta(seconds=5)}
client_agent3 = {'client_agent0': timedelta(seconds=2), 'client_agent1': timedelta(seconds=4), 'server_agent0': timedelta(seconds=1)}
server_agent1 = {'client_agent0': timedelta(seconds=3), 'client_agent1': timedelta(seconds=5), 'client_agent2': timedelta(seconds=1)}
LATENCY_DICT = {'client_agent0': client_agent1, 'client_agent1': client_agent2, 'client_agent2': client_agent3, 'server_agent0': server_agent1}
########################
tolerance = 0.75 # tolerance for weights convergance
epsilon = 0.1 #0.1, 0.5, 1.0, 10
mean = 0
########################
LOG_MAX_ITER = 10 # max iterations for the logistic regression
