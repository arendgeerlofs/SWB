import matplotlib.pyplot as plt
from dynsimf.models.Model import Model

from initialise import init_states, init_network
from parameters import constants, network_parameters
from update import update_conditions, update_SWB, update_SWB2, update_network, event


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

    # Update rules
    model.add_update(update_SWB2, {"model":model})
    # model.add_update(update_expectations, {"model": model})
    model.add_update(event, {"model": model},
                            condition = update_conditions["Event"], get_nodes=True) 
    if model.constants["upd_net"]:
        model.add_network_update(update_network, {"model":model}, get_nodes=True, condition = update_conditions["Network"])

    return model

def run_model(model, iterations, verbose=True):
    # Simulate model
    output = model.simulate(iterations)


    # Data visualisation and analysis

    # Print SWB scores over time of person 0
    SWB_scores = [[output["states"][a][0][0]] for a in output["states"]]
    fin_scores = [[output["states"][a][0][9]] for a in output["states"]]
    expectation_scores = [[output["states"][a][0][12]] for a in output["states"]]

    # Plot SWB scores over time
    # TODO change to averages
    # print(SWB_scores)
    plt.plot(fin_scores)
    plt.savefig("figures/test")
    plt.plot(expectation_scores)
    plt.savefig("figures/test2")

    return output

def visualise(model, output): 
    visualization_config = {
        'layout': 'fr',
        'plot_interval': 1,
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