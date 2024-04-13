import numpy as np

from dynsimf.models.Model import Model
from dynsimf.models.components.Update import Update
from dynsimf.models.components.Scheme import Scheme
from dynsimf.models.components.Update import UpdateType
from dynsimf.models.components.Update import UpdateConfiguration
from dynsimf.models.components.conditions.CustomCondition import CustomCondition

from initialise import init_states, init_network, initial_fin_hist, initial_RFC_hist, initial_SWB_hist, initial_nonfin_hist
from functions import get_nodes, calc_RFC
from update import update_conditions, update_states, update_network, event, initial_network_update, intervention, pulse, set_pulse, initial_RFC_update
import dask

def init_model(constants):
    # Create values for fin and nonfin that are used in SDA distance
    init_fin = np.random.uniform(constants["L_low"], constants["L_high"], constants['N'])
    init_nonfin = np.random.uniform(constants["L_low"], constants["L_high"], constants['N'])

    # Create network
    network = init_network(constants["N"], constants["type"],
                            constants["p"], constants["m"],
                            constants["segregation"], constants["beta"],
                            init_fin, init_nonfin)

    # Model configuration
    model = Model(network) # Initialize a model object
    model.constants = constants # Set the constants

    # Save init values in model
    model.init_fin = init_fin
    model.init_nonfin = init_nonfin

    model.set_states(list(init_states.keys())) # Add the states to the model
    # Initialization parameters
    init_params = {
        'model': model
    }
    model.set_initial_state(init_states, init_params)

    model.fin_hist = initial_fin_hist(model)
    model.nonfin_hist = initial_nonfin_hist(model)
    model.RFC_hist = initial_RFC_hist(model, model.get_state("RFC"))
    model.SWB_hist = initial_SWB_hist(model)

    # Iteration update

    # # Set SDA connections if network type is SDA
    # if model.constants["type"] == "SDA":
    #     initial_network_cfg = UpdateConfiguration({
    #         'arguments': {"model": model},
    #         'get_nodes': False,
    #         'update_type': UpdateType.NETWORK
    #         })
    #     int_net = Update(initial_network_update, initial_network_cfg) # Create an Update object that contains the object function
    #     model.add_scheme(Scheme(get_nodes, {'args': {'graph': model.graph}, 'lower_bound': 0, 'upper_bound': 1, 'updates': [int_net]}))

    #     initial_RFC_cfg = UpdateConfiguration({
    #         'arguments': {"model": model},
    #         'get_nodes': False,
    #         'update_type': UpdateType.STATE
    #         })
    #     int_RFC = Update(initial_RFC_update, initial_RFC_cfg)
    #     model.add_scheme(Scheme(get_nodes, {'args': {'graph': model.graph}, 'lower_bound': 0, 'upper_bound': 1, 'updates': [int_RFC]}))

    # Test
    # set_time_condition = CustomCondition(set_pulse, arguments=[model, 1])
    # set_time_condition2 = CustomCondition(set_pulse, arguments=[model, 2])
    # model.add_network_update(initial_network_update, {"model":model}, condition = set_time_condition)
    # model.add_update(initial_RFC_update, {"model":model}, condition = set_time_condition2)

    model.add_update(update_states, {"model":model})

    # Update rules
    model.add_update(event, {"model": model},
                           condition = update_conditions["Event"], get_nodes=True) 
    if model.constants["upd_net"]:
        model.add_network_update(update_network, {"model":model}, get_nodes=True, condition = update_conditions["Network"])
    
    # reoccuring intervention
    add_c = CustomCondition(pulse, arguments=[model])
    model.add_update(intervention, {"model": model}, condition=add_c)

    # Set intervention
    # TODO set intervention update

    return model

# @dask.delayed
def run_model(model, iterations, verbose=True):
    # Simulate model
    output = model.simulate(iterations, show_tqdm=verbose)

    return output
