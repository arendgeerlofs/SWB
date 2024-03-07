from dynsimf.models.components.conditions.Condition import ConditionType
from dynsimf.models.components.conditions.StochasticCondition import StochasticCondition

# Update conditions
update_conditions = {
    "SWB" : StochasticCondition(ConditionType.STATE, 1),
    "Event" : StochasticCondition(ConditionType.STATE, 0.1),
}

# Update functions
def update_SWB(model):
    fin = model.get_state("financial")
    fin_exp = model.get_state("fin_expected")
    SWB = model.get_state("SWB")
    # TODO SWB update
    delta = fin - fin_exp
    return {"SWB": SWB + delta}

def update_expectations(model):
    hab, fin, fin_exp = model.get_state("habituation"), model.get_state("financial"), model.get_state("fin_expected")
    # TODO rate of change
    # TODO is it possible to just call this function for the different expectations instead of explicitly doing them
    rate_of_change = hab / 10
    delta_fin = fin - fin_exp
    return {"fin_expected": fin_exp + delta_fin * rate_of_change}

def update_network(nodes):
    network_update = {}
    for node in nodes:
        continue
    return {'edge_change': network_update}


def event(nodes, model):
    fin = model.get_state("financial")
    event_size = model.constants["event_size"]
    change = event_size
    result = fin[nodes] + change
    return {"financial": result}

# TODO add interventions
