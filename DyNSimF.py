import matplotlib.pyplot as plt
from dynsimf.models.Model import Model

from initialise import *
from parameters import constants, network_parameters
from update import *


def model_func():
    # Create network
    Network = init_network(constants["N"], network_parameters["type"],
                           network_parameters["p"], network_parameters["m"])

    # Model configuration
    model = Model(Network) # Initialize a model object
    model.constants = constants # Set the constants
    model.set_states(list(init_states.keys())) # Add the states to the model

    # The paramaters we want to receive in our initalization functions
    init_params = {
        'constants': model.constants
    }
    model.set_initial_state(init_states, init_params)


    # Add expected states
    exp_states = {
        "fin_expected" : model.get_state("financial"),
    }
    model.set_states(list(init_states.keys()) + list(exp_states.keys()))

    
    # Update rules
    model.add_update(update_SWB, {"SWB":model.get_state("SWB"), 
                                "fin":model.get_state("financial"),
                                "fin_exp":model.get_state("fin_expected")},
                                condition = update_conditions["SWB"])
    # model.add_update(update_expectations, {"hab": model.get_state("habituation"),
    #                                        "fin": model.get_state("financial"),
    #                                        "fin_exp": model.get_state("fin_expected")})
    model.add_update(event, {"fin":model.get_state("financial"),
                            "event_size":model.constants['event_size']},
                            condition = update_conditions["Event"])
    model.add_network_update(update_network, get_nodes=True)

    return model


def run_model(model, iterations, verbose=True):
    # Simulate model
    output = model.simulate(iterations)


    # Data visualisation and analysis

    # Print SWB scores over time of person 0
    SWB_scores = [[output["states"][a][0][0]] for a in output["states"]]

    # Plot SWB scores over time
    # TODO change to averages
    plt.plot(SWB_scores)
    plt.savefig("figures/test")