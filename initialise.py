import networkx as nx
import numpy as np


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

def initial_SWB(constants):
    "Randomly initialise SWB using the normal distribution"
    return np.clip(np.random.normal(7, 2, constants['N']), 0, 10)

def initial_Likert(constants):
    "Randomly initialise Likert-type properties using uniform distribution"
    return np.random.uniform(constants["L_low"], constants["L_high"], constants['N'])

# TODO find way to remove variable constants from these functions
def initial_expected_fin(constants):
    "Initialise expected values based on actual initial value"
    return np.random.uniform(constants["L_low"], constants["L_high"], constants['N'])
    return model.get_state("financial")

def initial_expected_rel(constants):
    "Initialise expected values based on actual initial value"
    return np.random.uniform(constants["L_low"], constants["L_high"], constants['N'])
    return model.get_state("religion")


init_states = {
    # The goal
    'SWB': initial_SWB, 

    # Adaptation properties
    'habituation': initial_Likert,
    'sensitisation': initial_Likert,
    'desensitisation': initial_Likert,

    # Personality traits
    'extraversion': initial_Likert,
    'neuroticism': initial_Likert,
    'openness': initial_Likert,
    'conscientiesness': initial_Likert,
    'agreeableness': initial_Likert,

    # Individual properties
    'financial': initial_Likert,
    'religion': initial_Likert,

    # Expected values
    # 'fin_expected': initial_expected_fin,
    # 'rel_expected': initial_expected_rel,
}


