import numpy as np

from dynsimf.models.Model import Model
from initialise import init_states, init_network, initial_fin_hist, initial_RFC_hist, initial_SWB_hist, initial_nonfin_hist
from update import update_conditions, update_states, update_network
import dask
from visualisation import plot_avg
from functions import extract_data

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

    # Save init values in model so it can be added in the states during initialisation
    model.init_fin = init_fin
    model.init_nonfin = init_nonfin

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
    model.SWB_hist = initial_SWB_hist(model)


    # Update states
    model.add_update(update_states, {"model":model})

    # Update network
    if model.constants["upd_net"]:
        model.add_network_update(update_network, {"model":model}, get_nodes=True, condition = update_conditions["Network"])

    return model

# @dask.delayed
def run_model(model, iterations, verbose=True):
    # Simulate model
    output = model.simulate(iterations, show_tqdm=verbose)

    return output

def run(constants, iterations, verbose=True):
    model = init_model(constants)
    output = run_model(model, iterations, verbose)

    return output


def all_scenarios(params, scenarios, its, verbose=True, plot=True):
    for scenario_name in scenarios:
        for param_id, param in enumerate(scenarios[scenario_name]):
            params[param] = scenarios[scenario_name][param]
        output = run(params, its, verbose)
        if plot:
            plot_avg(output, name_addition=f"scenarios/{scenario_name}")
    return

