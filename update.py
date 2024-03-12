import numpy as np
from dynsimf.models.components.conditions.Condition import ConditionType
from dynsimf.models.components.conditions.StochasticCondition import StochasticCondition
from parameters import constants
from scipy.stats import rankdata

# Update conditions
update_conditions = {
    "SWB" : StochasticCondition(ConditionType.STATE, 1),
    "Event" : StochasticCondition(ConditionType.STATE, constants["event_prob"]),
    "Network" : StochasticCondition(ConditionType.STATE, 0.25),
    
}

# Update functions
def update_SWB(model, MF=True):
    hab, fin, fin_exp = model.get_state("habituation"), model.get_state("financial"), model.get_state("fin_expected")
    SWB = model.get_state("SWB")

    # Expectations update
    # TODO rate of change
    # TODO is it possible to just call this function for the different expectations instead of explicitly doing them
    rate_of_change = hab / 10
    delta_fin = fin - fin_exp
    if MF:
        for node in model.nodes:
            neighbors = model.get_neighbors(node)
            MF_score = np.mean(fin[neighbors])
            delta_fin[node] += (MF_score - fin[node])
        delta_fin /= 2

    # SWB update
    # TODO SWB update Diminishing returns
    delta = ((SWB - 10) * np.exp(-0.2*delta_fin) + (10 - SWB))
    return {"SWB": np.clip(SWB + delta, 0, 10), "RFC_expected": fin_exp + delta_fin * rate_of_change}

# """
def update_SWB2(model):
    hab, fin, RFC_exp = model.get_state("habituation"), model.get_state("financial"), model.get_state("RFC_expected")
    SWB = model.get_state("SWB")
    N = model.constants["N"]
    w = 0.5
    RFC_cur = np.empty(N)
    for node in model.nodes:
        I = fin[node]
        neighbors = model.get_neighbors(node)
        social_circle = np.append(neighbors, node)
        I_min = np.min(fin[social_circle])
        I_max = np.max(fin[social_circle])
        R_i = (I - I_min)/(I_max-I_min)
        F_i = rankdata(fin[social_circle])[-1]/len(social_circle)
        RFC_cur[node] = w * R_i + (1-w)*F_i
    RFC_delta = RFC_cur-RFC_exp


    # Change SWB based on Range-Frequency
    # TODO diminishing returns function over expectation and income
    SWB_change = ((SWB - 10) * np.exp(-0.2*RFC_delta) + (10 - SWB))

    # Change expectations
    # TODO habituation/sensitization
    rate_of_change = hab / 10
    exp_change = RFC_delta * rate_of_change
    return {"SWB": np.clip(SWB + SWB_change, 0, 10), "RFC_expected": RFC_exp + exp_change}




# """

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
