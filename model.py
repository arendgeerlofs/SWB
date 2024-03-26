from dynsimf.models.Model import Model
from dynsimf.models.components.Update import Update
from dynsimf.models.components.Scheme import Scheme
from dynsimf.models.components.Update import UpdateType
from dynsimf.models.components.Update import UpdateConfiguration
from dynsimf.models.components.conditions.CustomCondition import CustomCondition

from initialise import init_states, init_network, initial_fin_hist, initial_RFC_hist, initial_SWB_hist
from parameters import network_parameters
from functions import get_nodes
from update import update_conditions, update_states, update_network, event, initial_network_update, intervention, pulse


def init_model(constants):
    # Create network
    network = init_network(network_parameters["N"], network_parameters["type"],
                           network_parameters["p"], network_parameters["m"])

    # Model configuration
    model = Model(network) # Initialize a model object
    model.constants = constants # Set the constants
    model.set_states(list(init_states.keys())) # Add the states to the model
    # Initialization parameters
    init_params = {
        'model': model
    }
    model.set_initial_state(init_states, init_params)

    model.fin_hist = initial_fin_hist(model)
    model.RFC_hist = initial_RFC_hist(model)
    model.SWB_hist = initial_SWB_hist(model)

    # Set SDA connections if network type is SDA
    if network_parameters["type"] == "SDA":
        initial_network_cfg = UpdateConfiguration({
            'arguments': {"model": model},
            'get_nodes': False,
            'update_type': UpdateType.NETWORK
            })
        int_net = Update(initial_network_update, initial_network_cfg) # Create an Update object that contains the object function
        model.add_scheme(Scheme(get_nodes, {'args': {'graph': model.graph}, 'lower_bound': 0, 'upper_bound': 1, 'updates': [int_net]}))

    # Update rules
    model.add_update(update_states, {"model":model})
    model.add_update(event, {"model": model},
                           condition = update_conditions["Event"], get_nodes=True) 
    if model.constants["upd_net"]:
        model.add_network_update(update_network, {"model":model}, get_nodes=True, condition = update_conditions["Network"])
    
    # intervention
    add_c = CustomCondition(pulse, arguments=[model])
    model.add_update(intervention, {"model": model}, condition=add_c)

    return model

def run_model(model, iterations, verbose=True):
    # Simulate model
    output = model.simulate(iterations, show_tqdm=verbose)

    return output
