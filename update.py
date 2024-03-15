import numpy as np
from dynsimf.models.components.conditions.Condition import ConditionType
from dynsimf.models.components.conditions.StochasticCondition import StochasticCondition
from parameters import constants
from functions import calc_RFC, bisection, SDA_prob, SDA_root
from parameters import network_parameters

# Update conditions
update_conditions = {
    "SWB" : StochasticCondition(ConditionType.STATE, 1),
    "Event" : StochasticCondition(ConditionType.STATE, constants["event_prob"]),
    "Network" : StochasticCondition(ConditionType.STATE, 0.25),
}

# Initial updates to the model
def initial_network_update(model):
    """
    Add initial network connections using Social Distance Attachment
    - inputs -
    model : DyNSimF model object containing the graph, states and constants

    - outputs -
    dictionary containing the edge changes per node.
    Each node has an add list for which a connection is made in the network
    """
    # Get N, expected connections, alpha and financial status
    N = model.constants["N"]
    exp_con = N * network_parameters["p"]
    fin = model.get_state("financial").reshape(N, 1)
    alpha = network_parameters["segregation"]

    # Calculate euclidean distance between nodes based on financial status
    dist = np.abs(fin - fin.T)

    # Calculate beta using the bisection method to find the root of the SDA function
    beta_min = 0.1
    beta_max = 10
    beta = bisection(SDA_root, exp_con, N, dist, alpha, beta_min, beta_max, 0.01)

    # Calculate connection probabilities
    probs = SDA_prob(dist, alpha, beta)

    # Create adjacency matrix and convert it to dictionary containing node pairs for edges
    adj_mat = np.random.binomial(size=np.shape(probs), n=1, p=probs)
    network_update = {}
    for node, row in enumerate(adj_mat):
        network_update[node] = {"add": []}
        for neighbor, value in enumerate(row):
            if value == 1 and node != neighbor:
                network_update[node]["add"].append(neighbor)
    
    return {'edge_change': network_update}

def initial_RFC_update(model):
    """
    Function to set expected RFC to current RFC since this cannot be done during initialisation
    """
    return {"RFC_expected": calc_RFC(model)}

def update_states(model):
    """
    Function that updates the states each iteration
    Changes expectatations and SWB and saves current RFC
    """
    # Load states
    hab, RFC_exp = model.get_state("habituation"), model.get_state("RFC_expected")
    SWB, SWB_norm = model.get_state("SWB"), model.get_state("SWB_norm")

    # Dict to save param changes to
    param_chgs = {}

    # Calculate current RFC
    RFC_cur = calc_RFC(model)

    # Save current RFC
    param_chgs["RFC"] = RFC_cur

    # Calculate difference between expected and actual RFC
    RFC_delta = RFC_cur-RFC_exp

    # Change SWB based on Range-Frequency
    # TODO diminishing returns function over expectation and income
    SWB_change = ((SWB - 10) * np.exp(-0.2*RFC_delta) + (10 - SWB))
    param_chgs["SWB"] = np.clip(SWB_norm + SWB_change, 0, 10)

    # Change expectations
    # TODO habituation/sensitization
    rate_of_change = hab / 10
    exp_change = RFC_delta * rate_of_change
    param_chgs["RFC_expected"] = RFC_exp + exp_change

    return param_chgs

def update_network(nodes, model):
    """
    Function which changes the network by removing and adding connections
    """
    # TODO adjacency graphs
    network_update = {}
    for node in nodes:
        if node == 1:
            network_update[node] = {"remove": [], "add": []}
            neighbors = model.get_neighbors(node)
            other_nodes = np.delete(np.arange(model.constants["N"]), neighbors)
            network_update[node]["remove"].append(np.random.choice(neighbors))
            network_update[node]["add"].append(np.random.choice(other_nodes))
    return {'edge_change': network_update}

def event(nodes, model):
    """
    Functions which simulates the occurance of an event to the financial status of nodes
    """
    # Get financial status and event size
    fin = model.get_state("financial")
    event_size = model.constants["event_size"]

    # Calculate event
    change = np.random.normal(0, event_size)

    return {"financial": fin[nodes] + change}

# TODO add interventions
