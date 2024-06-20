from model import init_model, run_model
from visualisation import plot_avg, plot_stoch_param, plot_stoch_components, plot_var, two_var_heatmap
from functions import extract_data, init_ind_params, mean_chg
import numpy as np
import matplotlib.pyplot as plt

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
        for param in scenarios[scenario_name]:
            params[param] = scenarios[scenario_name][param]
        output = run_exec(params, its, verbose)
        if plot:
            plot_avg(output, name_addition=f"scenarios/{scenario_name}")
    return

def stoch_plot_param(constants, runs, its, param, bounds, samples, plot_components=False, title_add=""):
    """
    Runs multiple simulations varying a specified parameter within given bounds,
    and plots the results for the subjective well-being (SWB) over different runs.

    Parameters:
    - constants (dict): Configuration parameters for the model.
    - runs (int): Number of runs to average over for each parameter value.
    - its (int): Number of iterations to run the simulation.
    - param (str): The parameter to vary.
    - bounds (tuple): The lower and upper bounds for the parameter.
    - samples (int): The number of samples between the bounds.
    - plot_components (bool): Whether to plot the driver components of SWB

    Returns:
    - None
    """
    param_int_list = ["N", "hist_len"]
    init_fin, init_nonfin, init_SWB = init_ind_params(constants)
    param_samples = np.linspace(bounds[0], bounds[1], samples)

    # Create data arrays
    SWB_data = np.empty((samples, runs, its+1))
    exp_SWB_data = np.copy(SWB_data)
    fin_data = np.copy(SWB_data)
    exp_fin_data = np.copy(SWB_data)
    nonfin_data = np.copy(SWB_data)
    exp_nonfin_data = np.copy(SWB_data)
    RFC_data = np.copy(SWB_data)
    exp_RFC_data = np.copy(SWB_data)
    soccap_data = np.copy(SWB_data)
    exp_soccap_data = np.copy(SWB_data)
    
    intervention_timesteps = constants["int_ts"]
    int_var = constants["int_var"]
    
    for i, param_value in enumerate(param_samples):
        if param in param_int_list:
            param_value = int(param_value)
        constants[param] = param_value
        for j in range(runs):
            output = run_exec(constants, its, init_fin, init_nonfin, init_SWB)
            SWB = extract_data(output, 1)
            SWB_data[i, j] = np.mean(SWB, axis=1)
            exp_SWB = extract_data(output, 2)
            exp_SWB_data[i, j] = np.mean(exp_SWB, axis=1)
            if plot_components:
                fin = extract_data(output, 8)
                fin_data[i, j] = np.mean(fin, axis=1)
                exp_fin = extract_data(output, 9)
                exp_fin_data[i, j] = np.mean(exp_fin, axis=1)
                nonfin = extract_data(output, 10)
                nonfin_data[i, j] = np.mean(nonfin, axis=1)
                exp_nonfin = extract_data(output, 11)
                exp_nonfin_data[i, j] = np.mean(exp_nonfin, axis=1)
                RFC = extract_data(output, 14)
                RFC_data[i, j] = np.mean(RFC, axis=1)
                exp_RFC = extract_data(output, 15)
                exp_RFC_data[i, j] = np.mean(exp_RFC, axis=1)
                soccap = extract_data(output, 16)
                soccap_data[i, j] = np.mean(soccap, axis=1)
                exp_soccap = extract_data(output, 17)
                exp_soccap_data[i, j] = np.mean(exp_soccap, axis=1)
    
    plot_stoch_param(SWB_data, param, param_samples, intervention_timesteps, int_var, title_add=title_add)
    if plot_components:
        plot_stoch_components(fin_data, exp_fin_data, nonfin_data, exp_nonfin_data, RFC_data, exp_RFC_data, soccap_data, exp_soccap_data, param, param_samples, intervention_timesteps, int_var, title_add=title_add)
    return

def run_var_plot(constants, runs, its):
    init_fin, init_nonfin, init_SWB = init_ind_params(constants)
    data = np.empty((runs, its+1))

    intervention_timesteps = constants["int_ts"]
    int_var = constants["int_var"]

    for i in range(runs):
        output = run_exec(constants, its, init_fin, init_nonfin, init_SWB)
        SWB = extract_data(output, 1)
        data[i] = np.var(SWB, axis=1)
    plot_var(data, intervention_timesteps, int_var)
    return

def run_two_var_heatmap(constants, runs, its, samples, params, bounds, title_add = "", hist_gap_comb=False):

    param_int_list = ["N", "hist_len", "intervention_gap"]
    init_fin, init_nonfin, init_SWB = init_ind_params(constants)
    param_1_values = 2**np.linspace(bounds[0][0], bounds[0][1], samples[0]) * 0.5
    param_2_values = np.linspace(bounds[1][0], bounds[1][1], samples[1])
    # Create data arrays
    SWB_data = np.empty((samples[0], samples[1], runs))
    SWB_baseline = np.empty((samples[0], samples[1], runs))
    chg_data = np.empty((samples[0], samples[1], runs))

    for i, param_1 in enumerate(param_1_values):
        new_constants = constants.copy()
        if hist_gap_comb:
            new_constants["intervention_gap"] = param_1
            new_constants["hist_len"] = 32 * 0.5
        else:
            if params[0] in param_int_list:
                param_1 = int(param_1)
            new_constants[params[0]] = param_1
        for j, param_2 in enumerate(param_2_values):
            print(f"---{i}------{j}--")
            if params[1] in param_int_list:
                param_2 = int(param_2)
            new_constants[params[1]] = param_2
            for k in range(runs):
                output = run_exec(new_constants, its, init_fin, init_nonfin, init_SWB)
                SWB = extract_data(output, 1)
                SWB_data[i, j, k] = np.mean(np.mean(SWB, axis=1)[-new_constants["burn_in_period"]:])
                SWB_baseline[i, j, k] = np.mean(np.mean(SWB, axis=1)[:new_constants["burn_in_period"]])
                chg_data[i, j, k] = mean_chg(SWB, new_constants["burn_in_period"], per_agent=True)[0]
    if hist_gap_comb:
        title_add=f"{title_add}_hist_comb"
    np.save("data/heatmap_SWB_data", SWB_data)
    np.save("data/heatmap_SWB_baseline", SWB_baseline)
    np.save("data/heatmap_chg_data", chg_data)
    two_var_heatmap(SWB_data, SWB_baseline, params, 16 / param_1_values, param_2_values, title_add=f"{title_add}")
    two_var_heatmap(chg_data, SWB_baseline, params, 16 / param_1_values, param_2_values, title_add=f"_chg{title_add}", per_person=True)
    return
