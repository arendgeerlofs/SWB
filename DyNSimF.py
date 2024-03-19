import matplotlib.pyplot as plt
from dynsimf.models.Model import Model
from dynsimf.models.components.Update import Update
from dynsimf.models.components.Scheme import Scheme
from dynsimf.models.components.Update import UpdateType
from dynsimf.models.components.Update import UpdateConfiguration
from dynsimf.models.components.conditions.CustomCondition import CustomCondition
from dynsimf.models.tools.SA import SensitivityAnalysis
from dynsimf.models.tools.SA import SAConfiguration
import numpy as np
import SALib
from tqdm import tqdm
from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample
from SALib.test_functions import Ishigami
from functions import extract_data

from initialise import init_states, init_network
from parameters import network_parameters
from functions import get_nodes
from update import update_conditions, update_states, update_network, event, initial_network_update, initial_exp_update, intervention


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
    int_RFC = Update(initial_exp_update, initial_RFC_cfg) # Create an Update object that contains the object function
    model.add_scheme(Scheme(get_nodes, {'args': {'graph': model.graph}, 'lower_bound': 0, 'upper_bound': 1, 'updates': [int_RFC]}))


    # def pulse(model_input, test):
    #     return model_input[0]
    
    # add_c = CustomCondition(pulse, arguments=[model])
    # # condition_nb_a = ThresholdCondition(ConditionType.ADJACENCY, condition_threshold_nb_cfg, chained_condition=add_c)
    # # intervention
    # model.add_update(intervention, {"model": model}, condition=add_c)
    initial_RFC_cfg = UpdateConfiguration({
        'arguments': {"model": model},
        'get_nodes': False,
        'update_type': UpdateType.STATE
        })
    intervention_update = Update(intervention, initial_RFC_cfg) # Create an Update object that contains the object function
    for i in range(100):
        model.add_scheme(Scheme(get_nodes, {'args': {'graph': model.graph}, 'lower_bound': i*10, 'upper_bound': i*10+1, 'updates': [intervention_update]}))



    # Update rules
    model.add_update(update_states, {"model":model})
    model.add_update(event, {"model": model},
                           condition = update_conditions["Event"], get_nodes=True) 
    if model.constants["upd_net"]:
       model.add_network_update(update_network, {"model":model}, get_nodes=True, condition = update_conditions["Network"])

    return model

def run_model(model, iterations, verbose=True):
    # Simulate model
    output = model.simulate(iterations)

    return output

def GSA(constants):

    # Define the model inputs
    problem = {
        'num_vars': 2,
        'names': ['event_prob', 'event_size'],
        'bounds': [[0, 0.2],
                [0, 5]]
    }

    # Generate samples
    param_values = sample(problem, 4)
    
    data = np.empty(param_values.shape[0])
    for index, params in tqdm(enumerate(param_values)):
        constants["event_prob"] = params[0]
        constants["event_size"] = params[1]
        model = init_model(constants)
        output = model.simulate(3)
        SWB = extract_data(output, 0)
        data[index] = np.mean(SWB[-1])
    
    # Perform analysis
    Si = analyze(problem, data, print_to_console=True)

    # Print the first-order sensitivity indices
    print(Si['S1'])


    """
    initial_state = {"SWB":extract_data(output, 0)[0]}
    print(initial_state)
    cfg = SAConfiguration(
        {
            'bounds': {'event_prob': (0, 0.1)},
            'iterations': 10,
            'initial_state': initial_state,
            'initial_args': {},
            'n': 5,
            'second_order': True,

            'algorithm_input': 'states',
            'algorithm': lambda x, state_number: np.mean(extract_data(x, state_number)[-1]),
            'output_type': '',
            'algorithm_args': {"state_number": 0},
        }
    )

    sa = SensitivityAnalysis(cfg, model)
    analysis = sa.analyze_sensitivity()
    print(analysis)
    print("x1-x2:", analysis['SWB']['S2'][0,1])
    analysis['SWB'].plot()
    plt.show()
    """