import matplotlib.pyplot as plt
from dynsimf.models.Model import Model
from dynsimf.models.components.Update import Update
from dynsimf.models.components.Scheme import Scheme
from dynsimf.models.components.Update import UpdateType
from dynsimf.models.components.Update import UpdateConfiguration
import numpy as np

from initialise import init_states, init_network
from parameters import constants, network_parameters
from functions import get_nodes
from update import update_conditions, update_states, update_network, event, initial_network_update, initial_RFC_update


def init_model():
    # Create network
    network = init_network(constants["N"], network_parameters["type"],
                           network_parameters["p"], network_parameters["m"])

    # Model configuration
    model = Model(network) # Initialize a model object
    model.constants = constants # Set the constants
    model.set_states(list(init_states.keys())) # Add the states to the model

    # Initialization parameters
    init_params = {
        'constants': model.constants
    }
    model.set_initial_state(init_states, init_params)
    
    # Set SDA connections if network type is SDA
    if network_parameters["type"] == "SDA":
        initial_network_cfg = UpdateConfiguration({
            'arguments': {"model": model},
            'get_nodes': False,
            'update_type': UpdateType.NETWORK
            })
        int_net = Update(initial_network_update, initial_network_cfg) # Create an Update object that contains the object function
        model.add_scheme(Scheme(get_nodes, {'args': {'graph': model.graph}, 'lower_bound': 0, 'upper_bound': 1, 'updates': [int_net]}))

    # Set expectations equal to actual value at the start of the simulation
    initial_RFC_cfg = UpdateConfiguration({
        'arguments': {"model": model},
        'get_nodes': False,
        'update_type': UpdateType.STATE
        })
    int_RFC = Update(initial_RFC_update, initial_RFC_cfg) # Create an Update object that contains the object function
    model.add_scheme(Scheme(get_nodes, {'args': {'graph': model.graph}, 'lower_bound': 0, 'upper_bound': 1, 'updates': [int_RFC]}))

    # Update rules
    model.add_update(update_states, {"model":model})
    # model.add_update(update_expectations, {"model": model})
    model.add_update(event, {"model": model},
                           condition = update_conditions["Event"], get_nodes=True) 
    if model.constants["upd_net"]:
       model.add_network_update(update_network, {"model":model}, get_nodes=True, condition = update_conditions["Network"])

    return model

def run_model(model, iterations, verbose=True):
    # Simulate model
    output = model.simulate(iterations)

    return output

def visualise(model, output): 
    visualization_config = {
        'layout': 'fr',
        'plot_interval': 5,
        'plot_variable': 'SWB',
        'variable_limits': {
            'SWB': [0, 10],
        },
        'color_scale': 'RdBu',
        'show_plot': True,
        'plot_output': 'animations/SWB.gif',
        'plot_title': 'SWB',
    }

    model.configure_visualization(visualization_config, output)
    model.visualize('animation')

def plot(output):
    # Plot data
    # Print SWB scores over time of person 0
    SWB_scores = [[output["states"][a][0][0]] for a in output["states"]]

    fin_scores = [[output["states"][a][0][10]] for a in output["states"]]
    expectation_scores = [[output["states"][a][0][13]] for a in output["states"]]
    RFC = [[output["states"][a][0][14]] for a in output["states"]]

    # Plot SWB scores over time
    # TODO change to averages
    # print(SWB_scores)
    plt.plot(SWB_scores[2:])
    plt.savefig("figures/SWB")
    plt.clf()   # Clear figure
    plt.plot(fin_scores[2:])
    plt.plot(expectation_scores[2:])
    plt.savefig("figures/fin")
    plt.clf()
    plt.plot(RFC[2:])
    plt.plot(expectation_scores[2:])
    plt.savefig("figures/RFC")

