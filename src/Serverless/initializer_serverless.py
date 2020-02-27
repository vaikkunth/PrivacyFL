"""
NOTE: The Serverless directory was written as a proof of concept extension of our library
that involves no servers, with clients communicating with each other directly.
"""



import sys
sys.path.append('..')

import config
import datetime
import numpy as np
import pickle

from client_agent_serverless import ClientAgentServerless
from directory import Directory
from pyspark.sql import SparkSession
from sklearn.datasets import load_digits
from directory_serverless import Directory


from utils import data_formatting

import multiprocessing
from multiprocessing.pool import ThreadPool
from message import Message

from utils.print_config import print_config
from utils.model_evaluator import ModelEvaluator



def client_computation_caller(inp):
    client_instance, iteration, lock = inp
    __ = client_instance.produce_weights(iteration, lock)
    return None

class InitializerServerless:
    def __init__(self, num_clients, iterations):
        """
        Offline stage of simulation. Initializes clients and servers for iteration as well as gives each client its data.
        :param num_clients: number of clients to be use for simulation
        :param num_servers: number of servers to be use for simulation. Personalized coding required if greater than 1.
        :param iterations: number of iterations to run simulation for
        """
        global len_per_iteration
        if config.USING_PYSPARK:
            spark = SparkSession.builder.appName('SecureFederatedLearning').getOrCreate()  # initialize spark session
            spark.sparkContext.setLogLevel("ERROR")  # supress sparks messages

        digits = load_digits()  # using sklearn's MNIST dataset
        X, y = digits.data, digits.target

        X_train, X_test = X[:-config.LEN_TEST], X[-config.LEN_TEST:]
        y_train, y_test = y[:-config.LEN_TEST], y[-config.LEN_TEST:]

        # extract only amount that we require
        number_of_samples = 0
        for client_name in config.client_names:
            len_per_iteration = config.LENS_PER_ITERATION[client_name]
            number_of_samples += len_per_iteration * iterations

        X_train = X_train[:number_of_samples]
        y_train = y_train[:number_of_samples]

        client_to_datasets = data_formatting.partition_data(X_train, y_train, config.client_names, iterations,
                                                            config.LENS_PER_ITERATION, cumulative=config.USING_CUMULATIVE, pyspark=config.USING_PYSPARK)

        #print_config(len_per_iteration=config.LEN_PER_ITERATION)
        print('\n \n \nSTARTING SIMULATION \n \n \n')

        active_clients = {'client_agent' + str(i) for i in range(num_clients)}
        self.clients = {
            'client_agent' + str(i): ClientAgentServerless(agent_number=i,
                                                 train_datasets=client_to_datasets['client_agent' + str(i)],
                                                 evaluator=ModelEvaluator(X_test, y_test),
                                                 active_clients=active_clients) for i in
            range(num_clients)}  # initialize the agents

        # create directory with mappings from names to instances
        self.directory = Directory(clients=self.clients)

        for agent_name, agent in self.clients.items():
            agent.set_directory(self.directory)
            agent.initializations()

        # OFFLINE diffie-helman key exchange
        # NOTE: this is sequential in implementation, but simulated as occuring parallel
        if config.USE_SECURITY:
            key_exchange_start = datetime.datetime.now()  # measuring how long the python script takes
            max_latencies = []
            for client_name, client in self.clients.items():
                # not including logic of sending/receiving public keys in latency computation since it is nearly zero
                client.send_pubkeys()
                max_latency = max(config.LATENCY_DICT[client_name].values())
                max_latencies.append(max_latency)
            simulated_time = max(max_latencies)

            key_exchange_end = datetime.datetime.now()  # measuring runtime
            key_exchange_duration = key_exchange_end - key_exchange_start
            simulated_time += key_exchange_duration
            if config.SIMULATE_LATENCIES:
                print(
                    'Diffie-helman key exchange simulated duration: {}\nDiffie-helman key exchange real run-time: {}\n'.format(
                        simulated_time, key_exchange_duration))

            for client_name, client in self.clients.items():
                client.initialize_common_keys()


    def request_values(self, num_iterations):
        """
        Method invoked to start simulation. Prints out what clients have converged on what iteration.
        Also prints out accuracy for each client on each iteration (what weights would be if not for the simulation) and federated accuaracy.
        :param iters: number of iterations to run
        """
        for i in range(1, num_iterations+1):
            m = multiprocessing.Manager()
            lock = m.Lock()
            with ThreadPool(len(self.clients)) as calling_pool:
                args = []
                for client_instance in self.clients.values():
                    args.append((client_instance, i, lock))
                __ = calling_pool.map(client_computation_caller, args)


    def run_simulation(self, num_iterations):
        """
        Online stage of simulation.
        :param num_iterations: number of iterations to run
        :param server_agent_name: which server to use. Defaults to first server.
        """
        # ONLINE
        self.request_values(num_iterations)
        for client_name, client_agent in self.directory.clients.items():
            client_agent.final_statistics()

