import networkx as nx
import numpy as np
from functions import calc_RFC

# TODO social distance attachment
def init_network(size=100, net_type="Rd", p=0.5, m=5):
    if net_type == "BA":
        return nx.barabasi_albert_graph(size, m)
    elif net_type == "Rd":
        return nx.erdos_renyi_graph(size, p)
    elif net_type == "Pref":
        # TODO add social distance attachment model
        return
    else:
        print("Invalid network structure name")
        print("Options are 'BA', 'Rd' and 'Pref'")

def initial_SWB_norm(model):
    "Randomly initialise SWB using the normal distribution"
    return np.clip(np.random.normal(7, 2, model.constants['N']), 0, 10)

def initial_SWB(model):
    "Set initial SWB equal to norm"
    return model.get_state("SWB_norm")

def initial_Likert(model):
    "Randomly initialise Likert-type properties using uniform distribution"
    return np.random.uniform(model.constants["L_low"], model.constants["L_high"], model.constants['N'])

# TODO find way to remove variable constants from these functions
def initial_expected_fin(model):
    "Initialise expected values based on actual initial value"
    # return np.random.uniform(model.constants["L_low"], model.constants["L_high"], model.constants['N'])
    return model.get_state("financial")

def initial_RFC(model):
    return calc_RFC(model)

def initial_expected_SWB(model):
    "Initialise expected values based on actual initial value"
    # return np.random.uniform(constants["L_low"], constants["L_high"], constants['N'])
    return model.get_state("SWB")

def initial_expected_RFC(model):
    "Initialise expected values based on actual initial value"
    # return np.random.uniform(constants["L_low"], constants["L_high"], constants['N'])
    return model.get_state("RFC")

def initial_fin_hist(model):
    return model.get_state("financial").reshape(model.constants["N"], 1).repeat(model.constants["hist_len"], axis=1)

def initial_RFC_hist(model):
    return model.get_state("RFC").reshape(model.constants["N"], 1).repeat(model.constants["hist_len"], axis=1)

def initial_SWB_hist(model):
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
    'habituation': initial_Likert,
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


