import numpy as np

from dynsimf.models.Model import Model
from dynsimf.models.components.conditions.Condition import ConditionType
from dynsimf.models.components.conditions.StochasticCondition import StochasticCondition
from initialise import init_states, init_network, initial_fin_hist, initial_RFC_hist, initial_SWB_hist, initial_nonfin_hist, initial_soc_cap_hist
from update import update_states, update_network


def init_model(constants, init_fin=False, init_nonfin=False, init_SWB=False):
    # Create values for fin and nonfin that are used in SDA distance
    if not type(init_fin) is np.ndarray:
        init_fin = np.random.uniform(constants["L_low"], constants["L_high"], constants['N'])
    if not type(init_nonfin) is np.ndarray:
        init_nonfin = np.random.uniform(constants["L_low"], constants["L_high"], constants['N'])
    if not type(init_SWB) is np.ndarray:
        init_SWB = np.clip(np.random.normal(constants["SWB_mu"], constants["SWB_sd"], constants['N']), 0, 10)

    # Create network
    network = init_network(constants["N"], constants["type"],
                            constants["p"], constants["m"],
                            constants["segregation"], constants["beta"],
                            init_fin, init_nonfin, init_SWB)

    # Model configuration
    model = Model(network) # Initialize a model object
    model.constants = constants # Set the constants

    # Save init values in model so it can be added in the states during initialisation
    model.init_fin = init_fin
    model.init_nonfin = init_nonfin
    model.init_SWB = init_SWB

    model.set_states(list(init_states.keys())) # Add the states to the model
    # Initialization parameters
    init_params = {
        'model': model
    }
    model.set_initial_state(init_states, init_params)

    # Create initial history values
    model.fin_hist = initial_fin_hist(model)
    model.nonfin_hist = initial_nonfin_hist(model)
    model.RFC_hist = initial_RFC_hist(model, model.get_state("RFC"))
    model.soc_cap_hist = initial_soc_cap_hist(model, model.get_state("soc_cap"))
    model.SWB_hist = initial_SWB_hist(model)


    update_conditions = {
        "Network" : StochasticCondition(ConditionType.STATE, 0.1),
    }

    # Update states
    model.add_update(update_states, {"model":model})

    # Update network
    if model.constants["upd_net"]:
        model.add_network_update(update_network, {"model":model}, get_nodes=True, condition = update_conditions["Network"])

    return model

def run_model(model, iterations, verbose=True):
    # Simulate model
    output = model.simulate(iterations, show_tqdm=verbose)

    return output
