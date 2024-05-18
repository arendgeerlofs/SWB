from model import init_model, run_model
from visualisation import plot_avg, plot_stoch_param
from functions import extract_data, init_ind_params
import numpy as np

def run_exec(constants, iterations, init_fin=False, init_nonfin=False, init_SWB=False, verbose=True):
    """
    Initializes and runs the model simulation.

    Parameters:
    - constants (dict): Configuration parameters for the model.
    - iterations (int): Number of iterations to run the simulation.
    - init_fin (numpy.ndarray or bool): Initial financial values or False to generate them randomly.
    - init_nonfin (numpy.ndarray or bool): Initial non-financial values or False to generate them randomly.
    - init_SWB (numpy.ndarray or bool): Initial subjective well-being values or False to generate them randomly.
    - verbose (bool): Whether to display a progress bar during simulation.

    Returns:
    - output: The result of the simulation, typically a history of the states over iterations.
    """
    model = init_model(constants, init_fin, init_nonfin, init_SWB)
    output = run_model(model, iterations, verbose)
    return output

def all_scenarios(params, scenarios, its, verbose=True, plot=True):
    """
    Runs the model for all specified scenarios and optionally plots the results.

    Parameters:
    - params (dict): Initial parameters for the model.
    - scenarios (dict): Dictionary of scenarios, each containing parameter modifications.
    - its (int): Number of iterations to run the simulation.
    - verbose (bool): Whether to display a progress bar during simulation.
    - plot (bool): Whether to plot the average results for each scenario.

    Returns:
    - None
    """
    for scenario_name in scenarios:
        for param_id, param in enumerate(scenarios[scenario_name]):
            params[param] = scenarios[scenario_name][param]
        output = run_exec(params, its, verbose)
        if plot:
            plot_avg(output, name_addition=f"scenarios/{scenario_name}")
    return

def stoch_plot_param(constants, runs, its, param, bounds, steps):
    """
    Runs multiple simulations varying a specified parameter within given bounds,
    and plots the results for the subjective well-being (SWB) over different runs.

    Parameters:
    - constants (dict): Configuration parameters for the model.
    - runs (int): Number of runs to average over for each parameter value.
    - its (int): Number of iterations to run the simulation.
    - param (str): The parameter to vary.
    - bounds (tuple): The lower and upper bounds for the parameter.
    - steps (int): The number of steps between the bounds.

    Returns:
    - None
    """
    param_int_list = ["N", "hist_len"]
    init_fin, init_nonfin, init_SWB = init_ind_params(constants)
    SWB_data = np.empty((steps, runs, its+1))
    param_steps = np.linspace(bounds[0], bounds[1], steps)
    
    for i, param_value in enumerate(param_steps):
        if param in param_int_list:
            param_value = int(param_value)
        constants[param] = param_value
        for j in range(runs):
            output = run_exec(constants, its, init_fin, init_nonfin, init_SWB)
            SWB = extract_data(constants["N"], output, 1)
            SWB_data[i, j] = np.mean(SWB, axis=1)
    
    plot_stoch_param(SWB_data, param, param_steps)
    return
