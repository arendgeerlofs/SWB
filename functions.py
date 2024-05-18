import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from scipy.stats import rankdata

def calc_RFC(model):
    """
    Calculate the Range-Frequency Compromise (RFC) for each node in the network.

    Parameters:
    - model: The model object containing the current state and constants.

    Returns:
    - RFC_cur: An array of RFC values for each node.
    """
    N = model.constants["N"]
    fin = model.get_state("financial")
    soc_w = model.get_state("soc_w") / 10
    RFC_cur = np.empty(N)
    for node in model.nodes:
        I = fin[node]
        neighbors = model.get_neighbors(node)
        social_circle = np.append(neighbors, node).astype(int)
        I_min = np.min(fin[social_circle])
        I_max = np.max(fin[social_circle])
        if I_min == I_max:
            RFC_cur[node] = 0.5
        else:
            R_i = (I - I_min) / (I_max - I_min)
            F_i = rankdata(fin[social_circle])[-1] / len(social_circle)
            RFC_cur[node] = soc_w[node] * R_i + (1 - soc_w[node]) * F_i
    return 10 * RFC_cur

def distance(fin, nonfin, SWB):
    """
    Calculate the Euclidean distance matrix between nodes based on financial, non-financial, and SWB states.

    Parameters:
    - fin: Array of financial states.
    - nonfin: Array of non-financial states.
    - SWB: Array of SWB (Subjective Well-Being) states.

    Returns:
    - dist: A matrix of Euclidean distances between nodes.
    """
    character = np.column_stack((fin, nonfin, SWB))
    dist = cdist(character, character, metric='euclidean')
    return dist

def calc_soc_cap(model):
    """
    Calculate the social capital for each node in the network.

    Parameters:
    - model: The model object containing the current state and constants.

    Returns:
    - An array of social capital values for each node.
    """
    N = model.constants["N"]
    fin = model.get_state("financial").reshape(N, 1)
    nonfin = model.get_state("nonfin").reshape(N, 1)
    SWB = model.get_state("SWB").reshape(N, 1)
    dist = distance(fin, nonfin, SWB)
    likeness = 1 - normalize(dist, norm="max", axis=1)
    adj_mat = model.get_adjacency()
    return np.maximum(np.mean(likeness * adj_mat, axis=1), 0.0001)

def SDA_prob(dist, alpha, beta):
    """
    Calculate the probability of connection changes based on distance and model parameters.

    Parameters:
    - dist: A matrix of distances between nodes.
    - alpha: Segregation parameter.
    - beta: Model parameter.

    Returns:
    - A matrix of probabilities for connection changes.
    """
    return 1 / (1 + (beta ** (-1) * dist) ** alpha)

def extract_data(nodes, output, state_number):
    """
    Extract the data from the model output for all nodes and return as a 3D array.

    Parameters:
    - nodes: The number of nodes.
    - output: The model output containing state data.
    - state_number: The index of the state to extract.

    Returns:
    - A 3D array of extracted data.
    """
    data = np.zeros((len(output["states"]), nodes))
    for timestep in output["states"]:
        data[timestep] = output["states"][timestep][:, state_number]
    return data

def calc_sens(sens, sens_factor, desens_factor, event_change, mode="fin"):
    """
    Calculate the new sensitivity values based on event changes.

    Parameters:
    - sens: Array of current sensitivity values.
    - sens_factor: Array of sensitization factors.
    - desens_factor: Array of desensitization factors.
    - event_change: Array of event changes.
    - mode: The type of event ("fin" for financial, "nonfin" for non-financial).

    Returns:
    - An array of updated sensitivity values.
    """
    new_sens = np.empty(len(sens))
    for node, value in enumerate(sens):
        if mode == "fin":
            if event_change[node] > 0:
                new_sens[node] = value / (1 + ((sens_factor[node] * event_change[node]) / 10))
            else:
                new_sens[node] = value * (1 + (-(desens_factor[node] * event_change[node]) / 10))
        elif mode == "nonfin":
            if event_change[node] > 0:
                new_sens[node] = value * (1 + ((sens_factor[node] * event_change[node]) / 10))
            else:
                new_sens[node] = value / (1 + (-(desens_factor[node] * event_change[node]) / 10))
    return np.clip(new_sens, 0.25, 4)

def init_ind_params(constants):
    """
    Initialize individual parameters for the model.

    Parameters:
    - constants: A dictionary of constants used in the model.

    Returns:
    - init_fin: Initial financial states.
    - init_nonfin: Initial non-financial states.
    - init_SWB: Initial SWB (Subjective Well-Being) states.
    """
    init_fin = np.random.uniform(constants["L_low"], constants["L_high"], constants['N'])
    init_nonfin = np.random.uniform(constants["L_low"], constants["L_high"], constants['N'])
    init_SWB = np.clip(np.random.normal(constants["SWB_mu"], constants["SWB_sd"], constants['N']), 0, 10)
    return (init_fin, init_nonfin, init_SWB)
