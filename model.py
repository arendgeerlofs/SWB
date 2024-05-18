import numpy as np

from dynsimf.models.Model import Model
from dynsimf.models.components.conditions.Condition import ConditionType
from dynsimf.models.components.conditions.StochasticCondition import StochasticCondition
from initialise import init_states, init_network, initial_fin_hist, initial_RFC_hist, initial_SWB_hist, initial_nonfin_hist, initial_soc_cap_hist
from update import update_states, update_network


def init_model(constants, init_fin=False, init_nonfin=False, init_SWB=False):
    """
    Initializes a model with the given constants and optional initial values.

    Parameters:
    - constants (dict): Configuration parameters for the model including network parameters.
    - init_fin (numpy.ndarray or bool): Initial financial values or False to generate them randomly.
    - init_nonfin (numpy.ndarray or bool): Initial non-financial values or False to generate them randomly.
    - init_SWB (numpy.ndarray or bool): Initial subjective well-being values or False to generate them randomly.

    Returns:
    - Model: An initialized model object.
    """
    
    # Initialize financial and non-financial values if not provided
    if not isinstance(init_fin, np.ndarray):
        init_fin = np.random.uniform(constants["L_low"], constants["L_high"], constants['N'])
    if not isinstance(init_nonfin, np.ndarray):
        init_nonfin = np.random.uniform(constants["L_low"], constants["L_high"], constants['N'])
    if not isinstance(init_SWB, np.ndarray):
        init_SWB = np.clip(np.random.normal(constants["SWB_mu"], constants["SWB_sd"], constants['N']), 0, 10)
    
    # Initialize the network with the given parameters
    network = init_network(constants["N"], constants["type"],
                           constants["p"], constants["m"],
                           constants["segregation"], constants["beta"],
                           init_fin, init_nonfin, init_SWB)
    
    # Create a model object and set its constants
    model = Model(network)
    model.constants = constants

    # Store initial values in the model for state initialization
    model.init_fin = init_fin
    model.init_nonfin = init_nonfin
    model.init_SWB = init_SWB

    # Set the initial states of the model
    model.set_states(list(init_states.keys()))

    # Parameters for initializing the model state
    init_params = {
        'model': model
    }
    model.set_initial_state(init_states, init_params)

    # Create initial history values for different aspects of the model
    model.fin_hist = initial_fin_hist(model)
    model.nonfin_hist = initial_nonfin_hist(model)
    model.RFC_hist = initial_RFC_hist(model, model.get_state("RFC"))
    model.soc_cap_hist = initial_soc_cap_hist(model, model.get_state("soc_cap"))
    model.SWB_hist = initial_SWB_hist(model)

    # Define conditions for updating the network
    update_conditions = {
        "Network": StochasticCondition(ConditionType.STATE, model.constants["net_upd_freq"]),
    }

    # Add state update function to the model
    model.add_update(update_states, {"model": model})

    # Add network update function to the model if network updates are enabled
    if model.constants["upd_net"]:
        model.add_network_update(update_network, {"model": model}, get_nodes=True, condition=update_conditions["Network"])

    return model

def run_model(model, iterations, verbose=True):
    """
    Runs the model simulation for a specified number of iterations.

    Parameters:
    - model (Model): The model object to be simulated.
    - iterations (int): Number of iterations to run the simulation.
    - verbose (bool): Whether to display a progress bar (default is True).

    Returns:
    - output: The result of the simulation, typically a history of the states over iterations.
    """
    
    # Run the simulation for the given number of iterations
    output = model.simulate(iterations, show_tqdm=verbose)

    return output
