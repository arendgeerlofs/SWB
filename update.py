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
def initial_update(model):
    RFC = calc_RFC(model)
    N = model.constants["N"]
    exp_con = N * network_parameters["p"]
    fin = model.get_state("financial").reshape(N, 1)
    dist = np.abs(fin - fin.T)

    alpha = network_parameters["segregation"]
    beta = bisection(SDA_root, exp_con, N, dist, alpha, 0.1, 10, 0.01)

    probs = SDA_prob(dist, alpha, beta)
    print(probs)
    network_update = {}




    # for node in range(model.constants["N"]):


    #     network_update[node] = {"remove": [], "add": []}
    #     neighbors = model.get_neighbors(node)
    #     other_nodes = np.delete(np.arange(model.constants["N"]), neighbors)
    #     network_update[node]["remove"].append(np.random.choice(neighbors))
    #     network_update[node]["add"].append(np.random.choice(other_nodes))
    return {"RFC_expected": RFC, 'edge_change': network_update}



def update_SWB(model):
    hab, RFC_exp = model.get_state("habituation"), model.get_state("RFC_expected")
    SWB = model.get_state("SWB")
    RFC_cur = calc_RFC(model)
    RFC_delta = RFC_cur*10-RFC_exp

    # Change SWB based on Range-Frequency
    # TODO diminishing returns function over expectation and income
    SWB_change = ((SWB - 10) * np.exp(-0.2*RFC_delta) + (10 - SWB))

    # Change expectations
    # TODO habituation/sensitization
    rate_of_change = hab / 10
    exp_change = RFC_delta * rate_of_change
    return {"SWB": np.clip(SWB + SWB_change, 0, 10), "RFC_expected": RFC_exp + exp_change}

def update_network(nodes, model):
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
    fin = model.get_state("financial")
    event_size = model.constants["event_size"]
    change = np.random.normal(0, event_size)
    result = fin[nodes] + change
    return {"financial": result}

# TODO add interventions
