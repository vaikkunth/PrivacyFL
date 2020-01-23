import numpy as np
import config
from datetime import datetime
from multiprocessing.pool import ThreadPool

from agent import Agent


def client_caller_factory(directory):
    """
    Creates an instance of client_caller that will be used to parallelize task distribution
    :param directory: directory containing instances of the clients and servers in the simulation
    :type directory: defined in Directory.py
    :return: distributor function
    """

    def client_caller(inp):
        """
        Calls compute weights method of the respective client and returns the weights and intercepts
        :param inp: tuple containing client name and iteration
        :return: client weights and intercepts
        :rtype: tuple of numpy arrays
        """
        client_name, iteration = inp
        client_instance = directory.clients[client_name]
        return_message_body = client_instance.compute_weights(iteration=iteration)
        return return_message_body

    return client_caller


def client_returner_factory(directory):
    """
    Similar to client_caller factory except used to return weights
    """

    def client_returner(inp):
        """
        :param inp: input containting client_name, iteration, federated weights, and federated intercepts
        :type inp: tuple
        :return: Bool whether client's weight have converged
        """
        client_name, iteration, weights, intercepts, simulated_time = inp
        client_instance = directory.clients[client_name]
        converged = client_instance.receive_weights(iteration, weights, intercepts, simulated_time)
        return converged

    return client_returner


class ServerAgent(Agent):
    """ Server agent that averages (federated) weights and returns them to clients"""
    def __init__(self, agent_number):
        super(ServerAgent, self).__init__(agent_number=agent_number, agent_type='server_agent')
        self.averaged_weights = {}
        self.averaged_intercepts = {}

    def request_values(self, iters):
        """
        Method invoked to start simulation. Prints out what clients have converged on what iteration.
        Also prints out accuracy for each client on each iteration (what weights would be if not for the simulation) and federated accuaracy.
        :param iters: number of iterations to run
        """
        converged = {} # maps client names to iteration of convergence
        client_caller = client_caller_factory(self.directory)
        client_returner = client_returner_factory(self.directory)
        for i in range(iters):
            weights = {}
            intercepts = {}
            calling_pool = ThreadPool(len(self.directory.clients))
            args = [(client_name, i) for client_name in self.directory.clients]
            messages = calling_pool.map(client_caller, args)

            vals = [(message.body['weights'], message.body['intercepts']) for message in messages]
            simulated_communication_times = dict(zip(self.directory.clients, [message.body['simulated_time'] for message in messages]))
            slowest_client = max(simulated_communication_times, key=simulated_communication_times.get)
            max_time = simulated_communication_times[slowest_client] # simulated time it would take for server to receive all values


            server_logic_start = datetime.now()

            # add them to the weights_dictionary
            for client_name, return_vals in zip(self.directory.clients.keys(), vals):
                client_weights, client_intercepts = return_vals
                weights[client_name] = np.array(client_weights)
                intercepts[client_name] = np.array(client_intercepts)

            weights_np = np.array(list(weights.values()))  # the weights for this iteration!
            intercepts_np = np.array(list(intercepts.values()))

            try:
                averaged_weights = np.average(weights_np, axis=0)  # gets rid of security offsets
            except:
                raise ValueError('''DATA INSUFFICIENT: Some client does not have a sample from each class so dimension of weights is incorrect. Make
                                 train length per iteration larger for each client to avoid this issue''')

            averaged_intercepts = np.average(intercepts_np, axis=0)
            self.averaged_weights[i] = averaged_weights  ## averaged weights for this iteration!!
            self.averaged_intercepts[i] = averaged_intercepts

            # add time server logic takes
            server_logic_end = datetime.now()
            server_logic_time = server_logic_end - server_logic_start
            max_time += server_logic_time

            returning_pool = ThreadPool(len(self.directory.clients))
            args = [(client_name, i, averaged_weights, averaged_intercepts, max_time + config.LATENCY_DICT[self.name][client_name]) for client_name in self.directory.clients]
            return_messages = returning_pool.map(client_returner, args)
            for message in return_messages:
                if message.body['value'] == True and message.sender not in converged:
                    converged[message.sender] = i

        for client_name in self.directory.clients.keys():
            if client_name in converged:
                print('Client {} converged on iteration {}'.format(client_name, converged[client_name]))
            if client_name not in converged:
                print('Client {} never converged'.format(client_name))


