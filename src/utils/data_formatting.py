import numpy as np


def partition_data(X, y, client_names, num_iterations, lens_per_iteration, cumulative=False):
    """
    Function used to partition data to give to clients for simulation.
    :type X: numpy array
    :type y: numpy array
    :param client_names: list of all client agents' name
    :param num_iterations: number of iterations to partition data for (initially set in config.py)
    :type num_iterations: int
    :param lens_per_iteration: length of new dataset available each iteration for each client
    :type lens_per_iteration: dictionary
    :param cumulative: flag that indidicates where dataset creation should be cumulative. If True,
    the dataset at iteration i+1 contains all of the data from previous iterations as well.
    :type cumulative: bool
    :return: dictionary mapping each client by name to another dictionary which contains its dataset for each
    iteration mapped by ints (iteration).
    
    
    """
    client_datasets = {client_name: None for client_name in client_names}
    # partition each client its data
    last_index = 0  # where to start the next client's dataset from
    for client_name in client_names:
        datasets_i = {}  # datasets for client i
        len_per_iteration = lens_per_iteration[client_name]
        start_idx = last_index
        last_index += num_iterations * len_per_iteration  # where this client's datasets will end
        for j in range(1, num_iterations+1):
            if cumulative: # dataset gets bigger each iteraton
                end_indx = start_idx + len_per_iteration * j
            else:
                end_indx = start_idx + len_per_iteration # add the length per iteration
                

            #print('From {} to {}'.format(start_idx, end_indx))
            X_ij = X[start_idx:end_indx]
            y_ij = y[start_idx:end_indx]

            datasets_i[j] = (X_ij, y_ij)

            if not cumulative:
                start_idx = end_indx # move up start index

        client_datasets[client_name] = datasets_i
    return client_datasets

