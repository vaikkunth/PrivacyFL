import sys
sys.path.append('..')

import numpy as np
import config
from datetime import datetime
import multiprocessing
from multiprocessing.pool import ThreadPool
from utils.latency_helper import find_slowest_time
from agent import Agent
from message import Message

def client_computation_caller(inp):
    client_instance, message = inp
    return_message = client_instance.produce_weights(message=message)
    return return_message


def client_weights_returner(inp):
    client_instance, message = inp
    converged = client_instance.receive_weights(message)
    return converged


def client_agent_dropout_caller(inp):
    client_instance, message = inp
    __ = client_instance.remove_active_clients(message)
    return None


class ServerAgent(Agent):
    """ Server agent that averages (federated) weights and returns them to clients"""

    def __init__(self, agent_number):
        super(ServerAgent, self).__init__(agent_number=agent_number, agent_type='server_agent')
        self.averaged_weights = {}
        self.averaged_intercepts = {}

    def request_values(self, num_iterations):
        """
        Method invoked to start simulation. Prints out what clients have converged on what iteration.
        Also prints out accuracy for each client on each iteration (what weights would be if not for the simulation) and federated accuaracy.
        :param iters: number of iterations to run
        """
        converged = {}  # maps client names to iteration of convergence. Contains all inactive clients
        active_clients = set(self.directory.clients.keys())

        for i in range(1, num_iterations+1):
            weights = {}
            intercepts = {}

            m = multiprocessing.Manager()
            lock = m.Lock()
            with ThreadPool(len(active_clients)) as calling_pool:
                args = []
                for client_name in active_clients:
                    client_instance = self.directory.clients[client_name]
                    body = {'iteration': i, 'lock': lock, 'simulated_time':config.LATENCY_DICT[self.name][client_name]}
                    arg = Message(sender_name=self.name, recipient_name=client_name, body=body)
                    args.append((client_instance, arg))
                messages = calling_pool.map(client_computation_caller, args)

            server_logic_start = datetime.now()

            vals = {message.sender: (message.body['weights'], message.body['intercepts']) for message in messages}
            simulated_time = find_slowest_time(messages)

            # add them to the weights_dictionary
            for client_name, return_vals in vals.items():
                client_weights, client_intercepts = return_vals
                weights[client_name] = np.array(client_weights)
                intercepts[client_name] = np.array(client_intercepts)

            weights_np = list(weights.values()) # the weights for this iteration!
            intercepts_np = list(intercepts.values())

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
            simulated_time += server_logic_time

            with ThreadPool(len(active_clients)) as returning_pool:
                args = []
                for client_name in active_clients:
                    client_instance = self.directory.clients[client_name]
                    body = {'iteration': i, 'return_weights': averaged_weights,
                            'return_intercepts': averaged_intercepts,
                            'simulated_time': simulated_time + config.LATENCY_DICT[self.name][client_name]}
                    message = Message(sender_name=self.name, recipient_name=client_name, body=body)
                    args.append((client_instance, message))
                return_messages = returning_pool.map(client_weights_returner, args)

            simulated_time = find_slowest_time(return_messages)
            server_logic_start = datetime.now()
            clients_to_remove = set()
            for message in return_messages:
                if message.body['converged'] == True and message.sender not in converged:  # converging
                    converged[message.sender] = i  # iteration of convergence
                    clients_to_remove.add(message.sender)

            server_logic_end = datetime.now()
            server_logic_time = server_logic_end - server_logic_start
            simulated_time += server_logic_time

            if config.CLIENT_DROPOUT:
                # tell the clients which other clients have dropped out
                active_clients -= clients_to_remove
                if len(active_clients) < 2: # no point in continuing if don't have at least 2 clients
                    self.print_convergences(converged)
                    return
                with ThreadPool(len(active_clients)) as calling_pool:
                    args = []
                    for client_name in active_clients:
                        client_instance = self.directory.clients[client_name]
                        body = {'clients_to_remove': clients_to_remove, 'simulated_time': simulated_time + config.LATENCY_DICT[self.name][client_name], 'iteration':i}
                        message = Message(sender_name=self.name, recipient_name=client_name, body=body)
                        args.append((client_instance, message))
                    __ = calling_pool.map(client_agent_dropout_caller, args)


        # at end of all iterations
        self.print_convergences(converged)

    def print_convergences(self, converged):
        """
        Used to print out all the clients that have converged at the end of request values
        :param converged: dict of converged clients containing iteration of convergence
        :type converged: dict
        """

        for client_name in self.directory.clients.keys():
            if client_name in converged:
                print('Client {} converged on iteration {}'.format(client_name, converged[client_name]))
            if client_name not in converged:
                print('Client {} never converged'.format(client_name))

    def final_statistics(self):
        """
        USED FOR RESEARCH PURPOSES.
        """
        # for research purposes
        client_accs = []
        fed_acc = []
        for client_name, client_instance in self.directory.clients.items():
            fed_acc.append(list(client_instance.federated_accuracy.values()))
            client_accs.append(list(client_instance.personal_accuracy.values()))

        if config.CLIENT_DROPOUT:
            print('Federated accuracies are {}'.format(dict(zip(self.directory.clients, fed_acc))))
        else:
            client_accs = list(np.mean(client_accs, axis=0))
            fed_acc = list(np.mean(fed_acc, axis=0))
            print('Personal accuracy on final iteration is {}'.format(client_accs))
            print('Federated accuracy on final iteration is {}'.format(fed_acc))  # should all be the same if no dropout

