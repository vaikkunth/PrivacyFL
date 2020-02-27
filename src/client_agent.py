import copy
import numpy as np
import sys
import random
import config
import threading
from warnings import simplefilter
from datetime import datetime
from sklearn import metrics

from agent import Agent
from message import Message
from utils.dp_mechanisms import laplace
import utils.diffie_hellman as dh
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler

from pyspark.ml.classification import LogisticRegression

simplefilter(action='ignore', category=FutureWarning)

class ClientAgent(Agent):
    def __init__(self, agent_number, train_datasets, evaluator, active_clients):
        """
        Initializes an instance of client agent

        :param agent_number: id for agent
        :type agent_number: int
        :param train_datasets: dictionary mapping iteration to dataset for given iteration
        :type train_datasets: dictionary indexed by ints mapping to pyspark dataframes
        :param evaluator: evaluator instance used to evaluate new weights
        :type evaluator: evaluator, defined in parallelized.py
        :param active_clients: Clients currently in simulation. Will be updated if clients drop out
        """
        super(ClientAgent, self).__init__(agent_number=agent_number, agent_type="client_agent")

        self.train_datasets = train_datasets
        self.evaluator = evaluator
        self.active_clients = active_clients

        self.directory = None
        self.pubkeyList = None
        self.seckeyList = None
        self.otherkeyList = None
        self.commonkeyList = None
        self.seeds = None
        self.deltas = None

        self.computation_times = {}

        self.personal_weights = {}  # personal weights. Maps iteration (int) to weights (numpy array)
        self.personal_intercepts = {}

        self.weights_dp_noise = {}  # keyed by iteration; noise added at each iteration
        self.intercepts_dp_noise = {}

        self.federated_weights = {}  # averaged weights
        self.federated_intercepts = {}

        self.personal_accuracy = {}
        self.federated_accuracy = {}

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

    def produce_weights(self, message):
        """
        :param message: message containing information necessary to produce weights for the iteration
        :type message: Message
        :return: message containing weights with security and/or DP noise added, as specified in config.py
        :rtype: Message
        """
        start_time = datetime.now()
        body = message.body
        iteration, lock, simulated_time = body['iteration'], body['lock'], body['simulated_time']

        if iteration - 1 > len(self.train_datasets):  # iteration is indexed starting from 1
            raise (ValueError(
                'Not enough data to support a {}th iteration. Either change iteration data length in config.py or decrease amount of iterations.'.format(
                    iteration)))

        if config.USING_PYSPARK:
            weights, intercepts = self.compute_weights_pyspark(iteration)
        else:
            weights, intercepts = self.compute_weights_sklearn(iteration)

        self.personal_weights[iteration] = weights
        self.personal_intercepts[iteration] = intercepts

        # create copies of weights and intercepts since we may be adding to them
        final_weights, final_intercepts = copy.deepcopy(weights), copy.deepcopy(intercepts)

        if config.USE_DP_PRIVACY:
            lock.acquire()  # for random seed
            final_weights, final_intercepts = \
                self.add_noise(weights=weights, intercepts=intercepts, iteration=iteration)
            lock.release()

        if config.USE_SECURITY:  # adding security via diffie-helman offsets
            final_weights, final_intercepts = \
                self.add_security_offsets(weights=final_weights, intercepts=final_intercepts)
        end_time = datetime.now()
        computation_time = end_time - start_time
        self.computation_times[iteration] = computation_time
        # multiply latency by two: first the server has to request the value, then the client has to return it

        simulated_time += computation_time + config.LATENCY_DICT[self.name]['server_agent0']

        body = {'weights': final_weights, 'intercepts': final_intercepts, 'iter': iteration,
                'computation_time': computation_time, 'simulated_time': simulated_time}  # generate body

        return Message(sender_name=self.name, recipient_name=self.directory.server_agents, body=body)

    def compute_weights_pyspark(self, iteration):
        """
        Example of a function that would compute weights. This one uses PySpark to perform
        logistic regression. If using this function, the datasets should be cumulative, i.e.,
        the dataset in iteration i+1 should have the data from all previous iterations since the
        weights are trained from scratch.
        :return: weights and intercepts
        :rtype: numpy arrays
        """
        dataset = self.train_datasets[iteration]
        lr = LogisticRegression(maxIter=config.LOG_MAX_ITER)
        lrModel = lr.fit(dataset)

        weights = lrModel.coefficientMatrix.toArray()
        intercepts = lrModel.interceptVector
        return weights, intercepts

    def compute_weights_sklearn(self, iteration):
        """
        Example of a function that would compute weights. This one uses sklearn to perform
        logistic regression. If using this function, the datasets should not be cumulative, i.e.,
        the dataset in iteration i+1 should be completely new data since the training starts with the
        federated weights from the previous iteration. Note that if using a compute_weights function like this,
        the 'personal weights' are not created with only this clients dataset since it uses the *federated*
        weights from previous iterations, which include other clients data.
        :return: weights, intercepts
        :rtype: numpy arrays
        """
        X, y = self.train_datasets[iteration]

        lr = SGDClassifier(alpha=0.0001, loss="log", random_state=config.RANDOM_SEEDS[self.name][iteration])

        # Assign prev round coefficients
        if iteration > 1:
            federated_weights = copy.deepcopy(self.federated_weights[iteration - 1])
            federated_intercepts = copy.deepcopy(self.federated_intercepts[iteration - 1])
        else:
            federated_weights = None
            federated_intercepts = None

        lr.fit(X, y, coef_init=federated_weights, intercept_init=federated_intercepts)
        local_weights = lr.coef_
        local_intercepts = lr.intercept_

        return local_weights, local_intercepts

    def add_noise(self, weights, intercepts, iteration):
        """
        Adds differentially private noise to weights as specified by parameters in config.py.
        Also adds noise to intercepts if specified in intercepts.py.
        The sensitivity is computed using the size of the smallest dataset used by any client in this iteration.
        Note that modifications to add_noise might be necessary depending if you are using cumulative or non cumulative
        datasets.
        :return: weights, intercepts
        :rtype: numpy arrays
        """
        weights_shape = weights.shape
        weights_dp_noise = np.zeros(weights_shape)

        intercepts_shape = intercepts.shape
        intercepts_dp_noise = np.zeros(intercepts_shape)

        # generate DP parameters
        active_clients_lens = [config.LENS_PER_ITERATION[client_name] for client_name in self.active_clients]

        smallest_dataset = min(active_clients_lens)
        if config.USING_CUMULATIVE:
            smallest_dataset *= iteration

        sensitivity = 2 / (
                len(self.active_clients) * smallest_dataset * config.alpha)
        epsilon = config.EPSILONS[self.name]

        random.seed(config.RANDOM_SEEDS[self.name][iteration])
        # adding differentially private noise
        for i in range(weights_shape[0]):  # weights_modified is 2-D
            for j in range(weights_shape[1]):
                if config.DP_ALGORITHM == 'Laplace':
                    dp_noise = laplace(mean=config.mean, sensitivity=sensitivity, epsilon=epsilon)
                elif config.DP_ALGORITHM == 'Gamma':
                    scale = sensitivity / epsilon
                    num_clients = len(self.directory.clients)
                    dp_noise = random.gammavariate(1 / num_clients, scale) - random.gammavariate(1 / num_clients,
                                                                                                 scale)
                else:
                    raise AssertionError('Need to specify config.DP_ALGORITHM as Laplace or Gamma')
                weights_dp_noise[i][j] = dp_noise

        if config.INTERCEPTS_DP_NOISE:
            for i in range(intercepts_shape[0]):
                if config.DP_ALGORITHM == 'Laplace':
                    dp_noise = laplace(mean=config.mean, sensitivity=sensitivity, epsilon=epsilon)
                elif config.DP_ALGORITHM == 'Gamma':
                    scale = sensitivity / epsilon
                    num_clients = len(self.directory.clients)
                    dp_noise = random.gammavariate(1 / num_clients, scale) - random.gammavariate(1 / num_clients, scale)
                else:
                    raise AssertionError('Need to specify config.DP_ALGORITHM as Laplace or Gamma')
                intercepts_dp_noise[i] = dp_noise

        weights_with_noise = copy.deepcopy(weights)  # make a copy to not mutate weights
        intercepts_with_noise = copy.deepcopy(intercepts)

        self.weights_dp_noise[iteration] = weights_dp_noise
        weights_with_noise += weights_dp_noise
        self.intercepts_dp_noise[iteration] = intercepts_dp_noise
        intercepts_with_noise += intercepts_dp_noise
        return weights_with_noise, intercepts_with_noise

    def add_security_offsets(self, weights, intercepts):
        """
        Called if config.USE_SECURITY flag is on. Uses the offsets established by the diffie helman key exchange
        to mask weights and intercepts. Client i adds the offset established with Client j to the weights if i < j
        and otherwise subtracts it if i > j. If i = j, the client does not add anything since it does not have an offset
        with itself.
        :return: weights, intercepts
        :rtype: numpy array, numpy array
        """
        adding = True  # Controls flow of loop. When other agent number is greater, subtract offset instead of add it
        for agent_name, offset in self.deltas.items():  # dictionary but should be ordered since Python 3
            if agent_name == self.name:
                adding = False  # from here on out subtract offsets for next clients
            elif agent_name in self.active_clients:
                if adding == True:
                    weights += offset
                    intercepts += offset
                else:
                    weights -= offset
                    intercepts -= offset
            else:
                # client no longer in simulation so don't add offset
                pass
        self.update_deltas()  # update the deltas after using them

        return weights, intercepts

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
                random.seed(seed) # generate new seed
                seed = random.randint(-sys.maxsize, sys.maxsize)
                self.seeds[agent] = seed
                self.deltas[agent] = delta

    def receive_weights(self, message):
        """
        Called by server agent to return federated weights.
        :param message: message containing return weights and other necessary information
        :type message: Message
        :return: Message indicating whether client has converged in training this iteration, which only
        matters if config.CLIENT_DROPOUT is True.
        :rtype: Message
        """
        body = message.body
        iteration, return_weights, return_intercepts, simulated_time = body['iteration'], body['return_weights'], body[
            'return_intercepts'], body['simulated_time']

        return_weights = copy.deepcopy(return_weights)
        return_intercepts = copy.deepcopy(return_intercepts)

        if config.USE_DP_PRIVACY and config.SUBTRACT_DP_NOISE:
            # subtract your own DP noise
            return_weights -= self.weights_dp_noise[iteration] / len(self.active_clients)
            return_intercepts -= self.intercepts_dp_noise[iteration] / len(self.active_clients)

        self.federated_weights[iteration] = return_weights
        self.federated_intercepts[iteration] = return_intercepts

        personal_weights = self.personal_weights[iteration]
        personal_intercepts = self.personal_intercepts[iteration]

        converged = self.satisfactory_weights((personal_weights, personal_intercepts), (
            return_weights, return_intercepts))  # check whether weights have converged
        personal_accuracy = self.evaluator.accuracy(personal_weights, personal_intercepts)
        federated_accuracy = self.evaluator.accuracy(return_weights, return_intercepts)

        self.personal_accuracy[iteration] = personal_accuracy
        self.federated_accuracy[iteration] = federated_accuracy

        args = [self.name, iteration, personal_accuracy, federated_accuracy]
        iteration_report = 'Performance Metrics for {} on iteration {} \n' \
                           '------------------------------------------- \n' \
                           'Personal accuracy: {} \n' \
                           'Federated accuracy: {} \n' \

        if config.SIMULATE_LATENCIES:
            args.append(self.computation_times[iteration])
            iteration_report += 'Personal computation time: {} \n'

            args.append(simulated_time)
            iteration_report += 'Simulated time to receive federated weights: {} \n \n'

        if config.VERBOSITY:
            print(iteration_report.format(*args))

        msg = Message(sender_name=self.name, recipient_name=self.directory.server_agents,
                      body={'converged': converged,
                            'simulated_time': simulated_time + config.LATENCY_DICT[self.name]['server_agent0']})
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

    def remove_active_clients(self, message):
        """
        Method invoked by server agent when clients have dropped out.
        If another client has dropped out, this client needs to know that so that
        it knows now to add that security offset, and also to be able to dynamically compute
        the differential privacy parameters.
        :return: None
        """
        body = message.body
        clients_to_remove, simulated_time, iteration = body['clients_to_remove'], body['simulated_time'], body[
            'iteration']

        print('Simulated time for client {} to finish iteration {}: {}\n'.format(self.name, iteration, simulated_time))

        self.active_clients -= clients_to_remove
        return None