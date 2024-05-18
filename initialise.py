import networkx as nx
import numpy as np
from functions import calc_RFC, SDA_prob, distance, calc_soc_cap
from scipy.spatial.distance import cdist

# Initialize the network based on different types
def init_network(size=100, net_type="Rd", p=0.5, m=5, alpha=0.3, beta=0.5, fin=0, nonfin=0, SWB=0):
    """
    Initialize a network based on the specified type.
    
    Parameters:
    - size (int): Number of nodes in the network.
    - net_type (str): Type of network ('BA' for Barabási-Albert, 'Rd' for Erdős-Rényi, 'SDA' for Social Distance Attachment).
    - p (float): Probability for edge creation in Erdős-Rényi graph.
    - m (int): Number of edges to attach from a new node to existing nodes in Barabási-Albert graph.
    - alpha (float): Parameter for the Social Distance Attachment probability calculation.
    - beta (float): Parameter for the Social Distance Attachment probability calculation.
    - fin, nonfin, SWB: Attributes used for distance calculation in the Social Distance Attachment model.
    
    Returns:
    - NetworkX graph: Initialized network.
    """
    if net_type == "BA":
        return nx.barabasi_albert_graph(size, m)
    elif net_type == "Rd":
        return nx.erdos_renyi_graph(size, p)
    elif net_type == "SDA":
        g = nx.Graph()
        g.add_nodes_from([i for i in range(size)])

        dist = distance(fin, nonfin, SWB)
        probs = SDA_prob(dist, alpha, beta)

        adj_mat = np.random.binomial(size=np.shape(probs), n=1, p=probs)
        for node, row in enumerate(adj_mat):
            for neighbor, value in enumerate(row):
                if value == 1 and node != neighbor:
                    g.add_edge(node, neighbor)
        return g
    else:
        print("Invalid network structure name")
        print("Options are 'BA', 'Rd' and 'SDA'")

def initial_SWB_norm(model):
    "Set initial SWB equal to previously initialised values"
    return model.init_SWB

def initial_SWB(model):
    "Set initial SWB equal to previously initialised values"
    return model.init_SWB

def initial_hab(model):
    # return np.full(model.constants["N"], model.constants["hist_len"])
    "Set initial habituation, lower numbers mean faster habituation"
    return np.random.uniform(0, model.constants["hist_len"], model.constants['N'])

def initial_Likert(model):
    "Randomly initialise Likert-type properties using uniform distribution"
    return np.random.uniform(model.constants["L_low"], model.constants["L_high"], model.constants['N'])

def initial_expected_fin(model):
    "Initialise expected values based on actual initial value"
    return model.get_state("financial")

def initial_expected_nonfin(model):
    "Initialise expected values based on actual initial value"
    return model.get_state("nonfin")

def initial_RFC(model):
    "Set initial RFC to actual RFC based on the initial financial statuses and social connections"
    return calc_RFC(model)

def initial_expected_SWB(model):
    "Initialise expected values based on actual initial value"
    return model.get_state("SWB")

def initial_expected_RFC(model):
    "Initialise expected values based on actual initial value"
    return model.get_state("RFC")

def initial_expected_soc_cap(model):
    "Initialise expected values based on actual initial value"
    return model.get_state("soc_cap")

def initial_fin_hist(model):
    "Initialise history of financial stock equal to current stock for history length of time steps"
    return model.get_state("financial").reshape(model.constants["N"], 1).repeat(model.constants["hist_len"], axis=1)

def initial_nonfin_hist(model):
    "Initialise history of financial stock equal to current stock for history length of time steps"
    return model.get_state("nonfin").reshape(model.constants["N"], 1).repeat(model.constants["hist_len"], axis=1)

def initial_RFC_hist(model, RFC):
    "Initialise history of RFC stock equal to current stock for history length of time steps"
    return RFC.reshape(model.constants["N"], 1).repeat(model.constants["hist_len"], axis=1)

def initial_soc_cap_hist(model, soc_cap):
    "Initialise history of RFC stock equal to current stock for history length of time steps"
    return soc_cap.reshape(model.constants["N"], 1).repeat(model.constants["hist_len"], axis=1)

def initial_SWB_hist(model):
    "Initialise history of SWB equal to current stock for history length of time steps"
    return model.get_state("SWB").reshape(model.constants["N"], 1).repeat(model.constants["hist_len"], axis=1)

def initial_SWB_comm(model):
    "Initialise expected values based on actual initial value"
    # return np.random.uniform(constants["L_low"], constants["L_high"], constants['N'])
    return np.full(model.constants["N"], 7)

def initial_sens(model):
    "Initialise (de)sensitivity scalar"
    return np.ones(model.constants["N"])

def init_fin(model):
    "Initialise financial state equal to previously initialised values"
    return model.init_fin

def init_nonfin(model):
    "Initialise non_financial state equal to previously initialised values"
    return model.init_nonfin

def initial_soc_cap(model):
    "Initialise social capital using the social capital function"
    return calc_soc_cap(model)

init_states = {
    # The goal
    "SWB_norm" : initial_SWB_norm,
    'SWB': initial_SWB,
    'SWB_exp': initial_expected_SWB,
    "SWB_comm" : initial_SWB_comm,

    # Adaptation properties
    'habituation': initial_hab,
    'sensitisation': initial_Likert,
    'desensitisation': initial_Likert,

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
}



