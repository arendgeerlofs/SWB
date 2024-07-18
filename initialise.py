"""
Initialise functions and state dictionary for the SWB model
"""

import networkx as nx
import numpy as np
from functions import calc_RFC, SDA_prob, distance, calc_soc_cap

def init_network(size=100, net_type="Rd", p=0.5, m=5, alpha=0.3, beta=0.5, fin=0, nonfin=0, SWB=0):
    """
    Initialize a network based on the specified type.
    
    Parameters:
    - size (int): Number of nodes in the network.
    - net_type (str): Type of network ('BA' for Barabási-Albert, 'Rd' for Erdős-Rényi,
      'SDA' for Social Distance Attachment).
    - p (float): Probability for edge creation in Erdős-Rényi graph.
    - m (int): Number of edges to attach from a new node to existing nodes in Barabási-Albert graph.
    - alpha (float): Parameter for the Social Distance Attachment probability calculation.
    - beta (float): Parameter for the Social Distance Attachment probability calculation.
    - fin, nonfin, SWB: Attributes used for distance calculation in the SDA model.
    
    Returns:
    - NetworkX graph: Initialized network.
    """
    if net_type == "BA":
        return nx.barabasi_albert_graph(size, int(m))
    elif net_type == "Rd":
        return nx.erdos_renyi_graph(size, p)
    elif net_type == "SDA":
        seed = 105
        rng = np.random.default_rng(seed)
        g = nx.Graph()
        g.add_nodes_from([i for i in range(size)])

        dist = distance(fin, nonfin, SWB)
        probs = SDA_prob(dist, alpha, beta)
        adj_mat = rng.binomial(size=np.shape(probs), n=1, p=probs)
        for node, row in enumerate(adj_mat):
            for neighbor, value in enumerate(row):
                if value == 1 and node != neighbor:
                    g.add_edge(node, neighbor)
        return g
    else:
        print("Invalid network structure name")
        print("Options are 'BA', 'Rd' and 'SDA'")

def initial_SWB_norm(model):
    """
    Set initial SWB equal to previously initialised values.
    
    Parameters:
    - model: The model object containing the current state and constants.
    
    Returns:
    - Initial SWB values as set in the model.
    """
    return model.init_SWB

def initial_SWB(model):
    """
    Set initial SWB equal to previously initialised values.
    
    Parameters:
    - model: The model object containing the current state and constants.
    
    Returns:
    - Initial SWB values as set in the model.
    """
    return model.init_SWB

def initial_hab(model):
    """
    Set initial habituation, lower numbers mean faster habituation.
    
    Parameters:
    - model: The model object containing the current state and constants.
    
    Returns:
    - An array of initial habituation values for each node.
    """
    # return np.full(model.constants["N"], model.constants["hist_len"])
    return np.random.uniform(0, model.constants["hist_len"], model.constants['N'])

def initial_Likert(model):
    """
    Randomly initialise Likert-type properties using uniform distribution.
    
    Parameters:
    - model: The model object containing the current state and constants.
    
    Returns:
    - An array of initial Likert-type values for each node.
    """
    return np.random.uniform(model.constants["L_low"], model.constants["L_high"],
                              model.constants['N'])

def initial_expected_fin(model):
    """
    Initialise expected values based on actual initial value.
    
    Parameters:
    - model: The model object containing the current state and constants.
    
    Returns:
    - Initial financial state values.
    """
    return model.get_state("financial")

def initial_expected_nonfin(model):
    """
    Initialise expected values based on actual initial value.
    
    Parameters:
    - model: The model object containing the current state and constants.
    
    Returns:
    - Initial non-financial state values.
    """
    return model.get_state("nonfin")

def initial_RFC(model):
    """
    Set initial RFC to actual RFC based on the initial financial statuses and social connections.
    
    Parameters:
    - model: The model object containing the current state and constants.
    
    Returns:
    - Initial RFC values calculated from the model.
    """
    return calc_RFC(model)

def initial_expected_SWB(model):
    """
    Initialise expected values based on actual initial value.
    
    Parameters:
    - model: The model object containing the current state and constants.
    
    Returns:
    - Initial SWB state values.
    """
    return model.get_state("SWB")

def initial_expected_RFC(model):
    """
    Initialise expected values based on actual initial value.
    
    Parameters:
    - model: The model object containing the current state and constants.
    
    Returns:
    - Initial RFC state values.
    """
    return model.get_state("RFC")

def initial_expected_soc_cap(model):
    """
    Initialise expected values based on actual initial value.
    
    Parameters:
    - model: The model object containing the current state and constants.
    
    Returns:
    - Initial social capital values.
    """
    return model.get_state("soc_cap")

def initial_sensitivity(model):
    """
    Initialize the sensitivity values for each node in the network.

    Parameters:
    - model: The model object containing the current state and constants.

    Returns:
    - An array of sensitivity values for each node scaled between 0.5 and 2.
    """
    # Generate random values uniformly distributed between L_low and L_high for each node
    values = np.random.uniform(model.constants["L_low"], model.constants["L_high"],
                                model.constants['N'])

    # Calculate the base for scaling values to an exponential scale between 0.5 and 2
    base = (2 / 0.5) ** (1 / (10 - 0))

    # Scale the values to the exponential scale and return
    return 0.5 * base ** (values - 0)

def initial_fin_hist(model):
    """
    Initialise history of financial stock equal to current stock for history length of time steps.
    
    Parameters:
    - model: The model object containing the current state and constants.
    
    Returns:
    - A 2D array representing the history of financial states for each node.
    """
    fin = model.get_state("financial")
    return fin.reshape(model.constants["N"], 1).repeat(model.constants["hist_len"], axis=1)

def initial_nonfin_hist(model):
    """
    Initialise history of financial stock equal to current stock for history length of time steps.
    
    Parameters:
    - model: The model object containing the current state and constants.
    
    Returns:
    - A 2D array representing the history of non-financial states for each node.
    """
    nonfin = model.get_state("nonfin")
    return nonfin.reshape(model.constants["N"], 1).repeat(model.constants["hist_len"], axis=1)

def initial_RFC_hist(model, RFC):
    """
    Initialise history of RFC stock equal to current stock for history length of time steps.
    
    Parameters:
    - model: The model object containing the current state and constants.
    - RFC: The initial RFC values for each node.
    
    Returns:
    - A 2D array representing the history of RFC values for each node.
    """
    return RFC.reshape(model.constants["N"], 1).repeat(model.constants["hist_len"], axis=1)

def initial_soc_cap_hist(model, soc_cap):
    """
    Initialise history of RFC stock equal to current stock for history length of time steps.
    
    Parameters:
    - model: The model object containing the current state and constants.
    - soc_cap: The initial social capital values for each node.
    
    Returns:
    - A 2D array representing the history of social capital values for each node.
    """
    return soc_cap.reshape(model.constants["N"], 1).repeat(model.constants["hist_len"], axis=1)

def initial_SWB_hist(model):
    """
    Initialise history of SWB equal to current stock for history length of time steps.
    
    Parameters:
    - model: The model object containing the current state and constants.
    
    Returns:
    - A 2D array representing the history of SWB values for each node.
    """
    SWB = model.get_state("SWB")
    return SWB.reshape(model.constants["N"], 1).repeat(model.constants["hist_len"], axis=1)

# def initial_SWB_comm(model):
#     """
#     Initialise SWB communication values to a constant value of 7.
    
#     Parameters:
#     - model: The model object containing the current state and constants.
    
#     Returns:
#     - An array of SWB communication values for each node.
#     """
#     # return np.random.uniform(constants["L_low"], constants["L_high"], constants['N'])
#     return np.full(model.constants["N"], 7)

def initial_sens(model):
    """
    Initialise (de)sensitivity scalar to 1 for each node.
    
    Parameters:
    - model: The model object containing the current state and constants.
    
    Returns:
    - An array of sensitivity scalars for each node.
    """
    return np.ones(model.constants["N"])

def init_fin(model):
    """
    Initialise financial state equal to previously initialised values.
    
    Parameters:
    - model: The model object containing the current state and constants.
    
    Returns:
    - The initial financial values for each node.
    """
    return model.init_fin

def init_nonfin(model):
    """
    Initialise non-financial state equal to previously initialised values.
    
    Parameters:
    - model: The model object containing the current state and constants.
    
    Returns:
    - The initial non-financial values for each node.
    """
    return model.init_nonfin

def initial_soc_cap(model):
    """
    Initialise social capital using the social capital function.
    
    Parameters:
    - model: The model object containing the current state and constants.
    
    Returns:
    - Initial social capital values calculated from the model.
    """
    return calc_soc_cap(model)

def initial_degrees(model):
    """
    Initialize the degree of each node in the network.
    
    Parameters:
    - model: The model object containing the current state and constants.
    
    Returns:
    - An array of degrees for each node.
    """
    return np.sum(model.get_adjacency(), axis=1)

init_states = {
    # The goal
    "SWB_norm" : initial_SWB_norm,
    'SWB': initial_SWB,
    'SWB_exp': initial_expected_SWB,
    # "SWB_comm" : initial_SWB_comm,

    # Adaptation properties
    'habituation': initial_hab,
    'sensitisation': initial_sensitivity,
    'desensitisation': initial_sensitivity,

    # # Personality traits
    'soc_w': initial_Likert,

    # Individual properties
    'financial': init_fin,
    'fin_exp': initial_expected_fin,
    'nonfin' : init_nonfin,
    'nonfin_exp' : initial_expected_nonfin,

    # Sensitivity index
    'fin_sens':initial_sens,
    'nonfin_sens':initial_sens,

    # Social comparison
    'RFC': initial_RFC,
    'RFC_exp': initial_expected_RFC,

    # Social capital
    'soc_cap': initial_soc_cap,
    'soc_cap_exp': initial_expected_soc_cap,

    # Connections
    'degrees' : initial_degrees,
}
