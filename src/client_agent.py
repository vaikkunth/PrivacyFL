import copy
import numpy as np
import sys
import random
import config
from warnings import simplefilter
from datetime import datetime

from agent import Agent
from message import Message
from utils.dp_mechanisms import laplace, boundedLaplace, staircase, gaussian
import utils.diffie_hellman as dh

from pyspark.ml.classification import LogisticRegression

simplefilter(action='ignore', category=FutureWarning)


class ClientAgent(Agent):
    def __init__(self, agent_number, train_datasets, evaluator, sensitivity):
        """
        Initializes an instance of client agent

        :param agent_number: id for agent
        :type agent_number: int
        :param train_datasets: dictionary mapping iteration to dataset for given iteration
        :type train_datasets: dictionary indexed by ints mapping to pyspark dataframes
        :param evaluator: evaluator instance used to evaluate new weights
        :type evaluator: evaluator, defined in parallelized.py
        :param sensitivity: sensitivity for differential privacy as defined in DPmechanisms.py
        :type sensitivity: int
        """
        super(ClientAgent, self).__init__(agent_number=agent_number, agent_type="client_agent")

        self.train_datasets = train_datasets
        self.evaluator = evaluator
        self.sensitivity = sensitivity

        self.directory = None
        self.pubkeyList = None
        self.seckeyList = None
        self.otherkeyList = None
        self.commonkeyList = None
        self.seeds = None
        self.deltas = None

        self.computation_times = {}

        self.personal_weights = {}  # personal weights. Maps iteration (int) to weights (numpy array)
        self.federated_weights = {}  # averaged weights

        self.personal_intercepts = {}
        self.federated_intercepts = {}

    def initializations(self):
        """
        Preforms initializions that have to be done after initializing instance
        :return: None
        :rtype: None
        """
        assert (self.directory is not None)
        clients = self.directory.clients
        num_clients = len(clients)

        pubkeyList, seckeyList = dh.keygeneration(num_clients, self.agent_number)

        # note this works because dicts are ordered in Python 3.6+
        self.pubkeyList = dict(zip(clients.keys(), pubkeyList))
        self.seckeyList = dict(zip(clients.keys(), seckeyList))

        # these dictionaries will be populated after key exchange

        self.otherkeyList = {agent_name: None for agent_name, __ in clients.items()}
        self.otherkeyList[self.name] = 0  # set to zero for yourself!

        self.commonkeyList = {agent_name: None for agent_name, __ in clients.items()}
        self.commonkeyList[self.name] = 0

        self.seeds = {agent_name: None for agent_name, __ in clients.items()}
        self.seeds[self.name] = 0

        self.deltas = {agent_name: None for agent_name, __ in clients.items()}
        self.deltas[self.name] = 0

    def send_pubkeys(self):
        """
        Sends public keys to other clients in simulations as required by diffie-helman protocol.
        """
        for agent_name, agent in self.directory.clients.items():
            pubkey = self.pubkeyList[agent_name]  # retrieve pubkey for client we're sending to
            body = {'pubkey': pubkey}
            msg = Message(sender_name=self.name, recipient_name=agent_name, body=body)
            agent.receive_pubkey(msg)  # invoke method of receiving agent

    def receive_pubkey(self, message):
        """
        Receives public key from another client
        :param message: message containing pubkey from another client
        :type message: instance of Message defined in message.py
        """
        sender = message.sender
        body = message.body
        pubkey = body["pubkey"]
        self.otherkeyList[sender] = pubkey

    def initialize_common_keys(self):
        """
        Initializes common key list to be used as offsets for sending weights
        """
        pubkeyList = list(self.pubkeyList.values())
        seckeyList = list(self.seckeyList.values())
        otherkeyList = list(self.otherkeyList.values())
        commonkeyList = dh.keyexchange(len(self.directory.clients), self.agent_number, pubkeyList, seckeyList,
                                       otherkeyList)  # generates common keys
        for i, agent in enumerate(self.commonkeyList):
            self.commonkeyList[agent] = commonkeyList[i]

        self.update_deltas()  # this method generates seeds and deltas from the common keys

    def compute_weights(self, iteration):
        """
        Method invoked by server when request weights.
        :param iteration: iteration of simulation currently on
        :type iteration: int
        :return: Message containing weights with offset and differential privacy added if specified in config file
        :rtype: instance of Message as defined in Message.py
        """
        start_time = datetime.now()

        if iteration > len(self.train_datasets):
            raise (ValueError(
                'Not enough data to support a {}th iteration. Either change iteration data length in config.py or decrease amount of iterations.'.format(
                    iteration + 1)))

        dataset = self.train_datasets[iteration]

        lr = LogisticRegression(maxIter=config.LOG_MAX_ITER)
        lrModel = lr.fit(dataset)

        weights = lrModel.coefficientMatrix.toArray()
        intercepts = lrModel.interceptVector
        self.personal_weights[iteration] = weights
        self.personal_intercepts[iteration] = intercepts

        # preparing value to send to server by adding deltas and DP noise
        weights_modified = copy.deepcopy(weights)
        intercepts_modified = copy.deepcopy(intercepts)

        # adding differentially private noise
        if config.USE_DP_PRIVACY:
            for i, weight_vec in enumerate(weights_modified):  # weights_modified is 2-D
                for j, weight in enumerate(weight_vec):
                    dp_noise = laplace(mean=config.mean, sensitivity=self.sensitivity, epsilon=config.epsilon)
                    weights_modified[i][j] += dp_noise

            for i, weight in enumerate(intercepts_modified):
                dp_noise = laplace(mean=config.mean, sensitivity=self.sensitivity, epsilon=config.epsilon)
                intercepts_modified[i] += dp_noise

        if config.USE_SECURITY:  # adding security via diffie-helman offsets
            adding = True
            for agent, offset in self.deltas.items():
                if offset == 0:
                    adding = False
                else:
                    if adding == True:
                        weights_modified += offset
                        intercepts_modified += offset
                    else:
                        weights_modified -= offset
                        intercepts_modified -= offset
            self.update_deltas()  # update the deltas after using them

        end_time = datetime.now()
        computation_time = end_time - start_time
        self.computation_times[iteration] = computation_time
        # multiply latency by two: first the server has to request the value, then the client has to return it
        simulated_time = computation_time + 2 * config.LATENCY_DICT[self.name]['server_agent0']

        body = {'weights': weights_modified, 'intercepts': intercepts_modified, 'iter': iteration,
                'computation_time': computation_time, 'simulated_time': simulated_time}  # generate body

        return Message(sender_name=self.name, recipient_name=self.directory.server_agents, body=body)

    def update_deltas(self):
        """
        Updates commonkeyList. Called after each iteration to update values.
        """

        if None not in self.commonkeyList.values():  # if first time calling this function
            agents_and_seeds = self.commonkeyList.items()
            self.commonkeyList = self.commonkeyList.fromkeys(self.commonkeyList.keys(), None)
        else:
            # use exisitng seed to generate new seeds and offsets
            agents_and_seeds = self.seeds.items()

        for agent, seed in agents_and_seeds:
            # uses current seeds to generate new deltas and new seeds
            if agent != self.name:
                seed_b = bin(seed)  # cast to binary
                delta_b = seed_b[:20]
                delta = int(delta_b, 2)  # convert back to decimal from base 2

                seed_b = seed_b[20:]
                seed = int(seed_b, 2)
                random.seed(seed)
                seed = random.randint(-sys.maxsize, sys.maxsize)
                self.seeds[agent] = seed
                self.deltas[agent] = delta

    def receive_weights(self, iteration, return_weights, return_intercepts, simulated_time):
        """
        Called by server to return federated weights after each iteration
        :param iteration: iteration currently on
        :type iteration: int
        :param return_weights: federated weights
        :type return_weights: numpy array
        :param return_intercepts: federeated intercepts
        :type return_intercepts: numpy array
        :param simulated_time: simulated time it would take for the client to receive the federated weights
        :type datetime
        :return: Message indicating whether weights have converged
        :rtype: instance of Message defined in Message.py
        """
        # receives weights after an iteration
        self.federated_weights[iteration] = return_weights
        self.federated_intercepts[iteration] = return_intercepts

        # evaluate best personal model and best federated model

        # average weights through all iterations
        personal_weights_averaged = np.average(list(self.personal_weights.values()), axis=0)
        personal_intercepts_averaged = np.average(list(self.personal_intercepts.values()), axis=0)

        # average federated weights from all iterations
        federated_weights_averaged = np.average(list(self.federated_weights.values()), axis=0)
        federated_intercepts_averaged = np.average(list(self.federated_intercepts.values()), axis=0)

        satisfactory = self.satisfactory_weights((personal_weights_averaged, personal_intercepts_averaged), (
            federated_weights_averaged, federated_intercepts_averaged))  # check whether weights have converged

        personal_accuracy = self.evaluator.accuracy(personal_weights_averaged, personal_intercepts_averaged)
        federated_accuracy = self.evaluator.accuracy(federated_weights_averaged, federated_intercepts_averaged)

        args = (self.name, iteration, personal_accuracy, self.computation_times[iteration], federated_accuracy, simulated_time)
        print(
            'Performance Metrics for {} on iteration {} \n'
            '------------------------------------------- \n'
            'Personal accuracy: {} \n'
            'Personal computation time: {} \n'
            'Federated accuracy: {} \n'
            'Simulated time to receive federated weights: {} \n'.format(*args))

        msg = Message(sender_name=self.name, recipient_name=self.directory.server_agents,
                      body={'value': satisfactory})
        return msg

    def satisfactory_weights(self, personal, federated):
        """
        Private function to check convergence of weights
        :param personal: personal weights and person intercepts
        :type personal: tuple of numpy arrays
        :param federated: federated weights and federated intercepts
        :type federated: tuple of numpy arrays
        :return: True if converged.
        :rtype: Bool
        """
        personal_weights, personal_intercepts = personal
        federated_weights, federated_intercepts = federated

        weights_differences = np.abs(federated_weights - personal_weights)
        intercepts_differences = np.abs(federated_intercepts - personal_intercepts)
        return (weights_differences < config.tolerance).all() and (
                intercepts_differences < config.tolerance).all()  # check all weights are close enough
