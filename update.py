from dynsimf.models.Model import Model
from dynsimf.models.Model import ModelConfiguration
from dynsimf.models.components.conditions.Condition import ConditionType
from dynsimf.models.components.conditions.StochasticCondition import StochasticCondition

# Update conditions
update_conditions = {
    "SWB" : StochasticCondition(ConditionType.STATE, 1),
    "Event" : StochasticCondition(ConditionType.STATE, 1),
}

# Update functions
def update_SWB(SWB, fin, fin_exp):
    # TODO SWB update
    delta = fin - fin_exp
    return {"SWB": SWB + delta/fin_exp}

def update_expectations(hab, fin, fin_exp):
    # TODO rate of change
    # TODO is it possible to just call this function for the different expectations instead of explicitly doing them
    rate_of_change = hab / 10
    delta_fin = fin - fin_exp
    return {"fin_expected": fin_exp + delta_fin * rate_of_change}

def update_network(nodes):
    network_update = {}
    for node in nodes:
        print(node)
    return {'edge_change': network_update}


def event(fin, event_size):
    change = event_size
    return {"financial": fin + change}

# TODO add interventions
