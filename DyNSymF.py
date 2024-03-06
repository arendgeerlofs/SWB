import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from dynsimf.models.Model import Model
from dynsimf.models.Model import ModelConfiguration
from dynsimf.models.components.conditions.Condition import ConditionType
from dynsimf.models.components.conditions.StochasticCondition import StochasticCondition

# Constants
constants = {
    'N' : 100,
    'm' : 5,
    "L_low" : 0,
    "L_high" : 10,
    "SWB_high" : 10
}

# Test population and social structure
# TODO preferential treatment
social_structure = nx.barabasi_albert_graph(constants['N'], constants['m'])

def initial_SWB(constants):
    return np.clip(np.random.normal(7, 2, constants['N']), 0, 10)

def initial_adaptation(constants):
    return np.random.uniform(constants["L_low"], constants["L_high"], constants['N'])

def initial_ind_char(constants):
    return np.random.uniform(10, size=constants['N'])

# Define external char based on social structure
#def external_char(constants):


initial_states = {
    'SWB': initial_SWB, 
    'habituation': initial_adaptation,
    'sensitisation': initial_adaptation,
    'desensitisation': initial_adaptation,
    'extraversion': initial_ind_char,
    'neuroticism': initial_ind_char,
    'openness': initial_ind_char,
    'conscientiesness': initial_ind_char,
    'agreeableness': initial_ind_char,
    'financial': initial_ind_char,
    'religion': initial_ind_char,
    'fin_expected': initial_ind_char,
}

# Model configuration
model = Model(social_structure) # Initialize a model object
model.constants = constants # Set the constants
model.set_states(list(initial_states.keys())) # Add the states to the model

# The paramaters we want to receive in our initalization functions
initial_params = {
    'constants': model.constants
}
model.set_initial_state(initial_states, initial_params)

# Update condition
s = StochasticCondition(ConditionType.STATE, 1)

# Update functions
def update_SWB():
    fin = model.get_state("financial")
    rel = model.get_state("religion")
    return {"SWB": model.get_state("SWB") / 1.1}

def update_expectations(constants):
    rate_of_change = model.get_state("habituation") / 10
    fin = model.get_state("financial")
    fin_expected = model.get_state("fin_expected")
    delta_fin = fin - fin_expected
    return {"fin_expected": fin_expected + delta_fin * rate_of_change}


# Rules
model.add_update(update_SWB, condition=s)

output = model.simulate(100)
SWB_scores = [[output["states"][a][0][0]] for a in output["states"]]
plt.plot(SWB_scores)
plt.savefig("figures/test")