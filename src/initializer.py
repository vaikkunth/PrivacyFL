import config
import datetime

from client_agent import ClientAgent
from server_agent import ServerAgent
from directory import Directory
from pyspark.sql import SparkSession
from sklearn.datasets import load_digits

from utils.data_formatting import create_spark_df
from utils.print_config import print_config
from utils.model_evaluator import ModelEvaluator


class Initializer:
    def __init__(self, num_clients, num_servers, iterations):
        """
        Offline stage of simulation. Initializes clients and servers for iteration as well as gives each client its data.
        :param num_clients: number of clients to be use for simulation
        :param num_servers: number of servers to be use for simulation. Personalized coding required if greater than 1.
        :param iterations: number of iterations to run simulation for
        """
        global len_per_iteration
        spark = SparkSession.builder.appName('SecureFederatedLearning').getOrCreate()  # initialize spark session
        spark.sparkContext.setLogLevel("ERROR")  # supress sparks messages

        digits = load_digits()  # using sklearn's MNIST dataset
        X, y = digits.data, digits.target

        X_train, X_test = X[:-config.LEN_TEST], X[-config.LEN_TEST:]
        y_train, y_test = y[:-config.LEN_TEST], y[-config.LEN_TEST:]
        df = create_spark_df(X_train, y_train)
        length_train_per_client = len(
            X_train) * 1 / num_clients  # how many samples will be used by each client for each iteration
        train_datasets = df.randomSplit([length_train_per_client] * num_clients)  # split up dataset for clients
        client_to_datasets = {}  # do the splitting by iterations now since clients cannot split the datasets dynamically very easily
        for i, dataset in enumerate(train_datasets):
            len_per_iteration = length_train_per_client / iterations
            client_datasets = dataset.randomSplit([len_per_iteration] * iterations)
            client_to_datasets[i] = client_datasets

        sensitivity = 2 / (num_clients * len_per_iteration)

        print_config(len_per_iteration=len_per_iteration, sensitivity=sensitivity)
        print('\n \n \nSTARTING SIMULATION \n \n \n')

        evaluator = ModelEvaluator(X_test, y_test)  # to evaluate new weights/intercepts later in the simulation

        self.clients = {
            'client_agent' + str(i): ClientAgent(agent_number=i,
                                                 train_datasets=client_to_datasets[i],
                                                 evaluator=evaluator,
                                                 sensitivity=sensitivity) for i in
            range(num_clients)}  # initialize the agents

        self.server_agents = {'server_agent' + str(i): ServerAgent(agent_number=i) for i in
                              range(num_servers)}  # initialize servers

        # create directory with mappings from names to instances
        self.directory = Directory(clients=self.clients, server_agents=self.server_agents)

        for agent_name, agent in self.clients.items():
            agent.set_directory(self.directory)
            agent.initializations()
        for agent_name, agent in self.server_agents.items():
            agent.set_directory(self.directory)

        # OFFLINE diffie-helman key exchange
        # NOTE: this is sequential in implementation, but simulated as occuring parallel

        key_exchange_start = datetime.datetime.now()  # measuring how long the python script takes

        max_latency_all_clients = datetime.timedelta(seconds=0)
        for client_name, client in self.clients.items():
            # not including logic of sending/receiving public keys in latency computation since it is nearly zero
            client.send_pubkeys()
            max_latency = datetime.timedelta(seconds=0)
            for latency in config.LATENCY_DICT[client_name].values():
                if latency > max_latency:
                    max_latency = latency

            if max_latency > max_latency_all_clients:
                max_latency_all_clients = max_latency

        key_exchange_end = datetime.datetime.now() # measuring runtime

        print(
            'Diffie-helman key exchange simulatedduration: {}\nDiffie-helman key exchange real run-time: {}\n'.format(
                max_latency_all_clients, key_exchange_end - key_exchange_start))

        for client_name, client in self.clients.items():
            client.initialize_common_keys()

    def run_simulation(self, num_iterations, server_agent_name='server_agent0'):
        """
        Online stage of simulation.
        :param num_iterations: number of iterations to run
        :param server_agent_name: which server to use. Defaults to first server.
        """
        # ONLINE
        server_agent = self.directory.server_agents[server_agent_name]
        server_agent.request_values(iters=num_iterations)
