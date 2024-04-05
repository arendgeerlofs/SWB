import networkx as nx
import numpy as np
from functions import calc_RFC

# TODO social distance attachment
def init_network(size=100, net_type="Rd", p=0.5, m=5):
    if net_type == "BA":
        return nx.barabasi_albert_graph(size, m)
    elif net_type == "Rd":
        return nx.erdos_renyi_graph(size, p)
    elif net_type == "SDA":
        g = nx.Graph()
        g.add_nodes_from([i for i in range(100)])
        return g
    else:
        print("Invalid network structure name")
        print("Options are 'BA', 'Rd' and 'SDA'")

def initial_SWB_norm(model):
    "Randomly initialise SWB using the normal distribution"
    return np.clip(np.random.normal(model.constants["SWB_mu"], model.constants["SWB_sd"], model.constants['N']), 0, 10)

def initial_SWB(model):
    "Set initial SWB equal to norm"
    return model.get_state("SWB_norm")

def initial_hab(model):
    "Set initial habituation, lower numbers mean faster habituation"
    return np.random.uniform(0, model.constants["hist_len"], model.constants['N'])


def initial_Likert(model):
    "Randomly initialise Likert-type properties using uniform distribution"
    return np.random.uniform(model.constants["L_low"], model.constants["L_high"], model.constants['N'])

def initial_expected_fin(model):
    "Initialise expected values based on actual initial value"
    return model.get_state("financial")

def initial_RFC(model):
    "Set initial RFC to actual RFC based on the initial financial statuses and social connections"
    return calc_RFC(model)

def initial_expected_SWB(model):
    "Initialise expected values based on actual initial value"
    return model.get_state("SWB")

def initial_expected_RFC(model):
    "Initialise expected values based on actual initial value"
    return model.get_state("RFC")

def initial_fin_hist(model):
    "Initialise history of financial stock equal to current stock for history length of time steps"
    return model.get_state("financial").reshape(model.constants["N"], 1).repeat(model.constants["hist_len"], axis=1)

def initial_RFC_hist(model):
    "Initialise history of RFC stock equal to current stock for history length of time steps"
    return model.get_state("RFC").reshape(model.constants["N"], 1).repeat(model.constants["hist_len"], axis=1)

def initial_SWB_hist(model):
    "Initialise history of SWB equal to current stock for history length of time steps"
    return model.get_state("SWB").reshape(model.constants["N"], 1).repeat(model.constants["hist_len"], axis=1)

def initial_SWB_comm(model):
    "Initialise expected values based on actual initial value"
    # return np.random.uniform(constants["L_low"], constants["L_high"], constants['N'])
    return model.get_state("RFC")


init_states = {
    # The goal
    "SWB_norm" : initial_SWB_norm,
    'SWB': initial_SWB,
    'SWB_exp': initial_expected_SWB,
    "SWB_comm" : initial_SWB_comm,

    # Adaptation properties
    'habituation': initial_hab,
    # 'sensitisation': initial_Likert,
    # 'desensitisation': initial_Likert,

    # # Personality traits
    # 'extraversion': initial_Likert,
    # 'neuroticism': initial_Likert,
    # 'openness': initial_Likert,
    # 'conscientiesness': initial_Likert,
    # 'agreeableness': initial_Likert,

    # Individual properties
    'financial': initial_Likert,
    # 'fin_hist' : initial_fin_hist,
    'fin_exp': initial_expected_fin,

    # External property
    # 'Evironment': initial_Likert,

    # Expected values
    'RFC': initial_RFC,
    # 'RFC_hist' : initial_RFC_hist,
    'RFC_exp': initial_expected_RFC,
}


