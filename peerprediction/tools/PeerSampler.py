# This code allows sampling peers for estimating rewards

import numpy as np

# IID sampling from Bernoulli process
# parameters: beliefs - 1-dimensional ndarray
#             num_samp - number of peers to sample
# Returns:    ndarray of dimensionality num_samp * beliefs.shape[0]
def sample_peers(beliefs, num_samp):
    num_tasks = beliefs.shape[0]
    peers = np.random.binomial(1, p=beliefs, size=(num_samp, num_tasks))
    return peers

# Expands original dataset to the required number of sampled peers
def expand_original_dataset(dataset, size):
    n_agents = dataset.shape[0]

    if n_agents == size:
        return dataset
    else:
        times_to_stack = int(size/n_agents)
        first = np.tile(dataset, (times_to_stack, 1))
        remainder = size % n_agents

        selection = np.random.choice(n_agents, size=remainder, replace=False)
        second = dataset[selection, :]
        shuffling = np.random.choice(size, size=size, replace=False)

        final_array = np.vstack((first, second))[shuffling, :]
        return final_array

# Flips reports in expanded dataset with some probability to converge to the beliefs of the agent
def modify_dataset(dataset, beliefs):
    num_agents = dataset.shape[0]
    num_tasks = dataset.shape[1]

    d_means = np.mean(dataset, axis=0)
    to_change = ((beliefs - d_means) < 0).astype(int)

    # Avoid dividing by 0
    temp_var = 1 - to_change - d_means
    temp_var[np.where(temp_var == 0)] = 0.01

    premask = (beliefs - d_means) / temp_var
    mask = (to_change == dataset).astype(int)

    dummy = np.random.binomial(1, p=premask, size=(num_agents, num_tasks))

    selection = (dummy * mask).astype(bool)
    output = np.copy(dataset)
    output[selection] = 1 - output[selection]

    return output

# Smart sampling using original dataset
# parameters: beliefs - 1-dimensional ndarray
#             num_samp - number of peers to sample
#             dataset - original dataset or reports, dataset.shape[1] should equal to beliefs.shape[0]
# Returns:    ndarray of dimensionality num_samp * beliefs.shape[0]
def smart1(beliefs, num_samp, dataset):
    expanded = expand_original_dataset(dataset, num_samp)
    output = modify_dataset(expanded, beliefs)
    return output

# DEPRECATED FUNCITONS BELOW

# def smart2(beliefs, num_samp, n_splits=10):
#     num_tasks = beliefs.shape[0]
#
#     iid_matrix = sample_peers(beliefs, num_samp)
#     subarrays = np.split(iid_matrix, n_splits)
#
#     for subarray in subarrays:
#         # print(subarray)
#         subarray = sorting_helper(subarray)
#         # print(subarray)
#
#     subarrays = np.array(subarrays)
#     # print(subarrays.shape)
#     subarrays = np.reshape(subarrays, (num_samp, num_tasks))
#     # print(subarrays.shape)
#
#     return subarrays
#
# def sorting_helper(array):
#     n_rows = array.shape[0]
#     s_id = np.random.choice(n_rows, 1, replace=False)
#     # print(s_id)
#     truncated = np.delete(array, s_id, axis=0)
#     equality_mask = (truncated == array[s_id, :]).astype(int)
#     sorted_trunc = ((-1 * np.sort(-1 * equality_mask, axis=0)) == array[s_id, :]).astype(int)
#     sorted_array = np.vstack((array[s_id, :], sorted_trunc))
#
#     shuffling = np.random.choice(n_rows, n_rows, replace=False)
#     sorted_array = sorted_array[shuffling, :]
#
#     return sorted_array
