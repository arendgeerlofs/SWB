from model import init_model, run_model
from visualisation import plot_agent, plot_avg, plot_stoch_param, plot_stoch_components, plot_var, two_var_heatmap, hist_plot, degree_plot
from functions import extract_data, init_ind_params, mean_chg
import numpy as np
import pandas as pd

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

# def all_scenarios(params, scenarios, its, verbose=True, plot=True):
#     """
#     Runs the model for all specified scenarios and optionally plots the results.

#     Parameters:
#     - params (dict): Initial parameters for the model.
#     - scenarios (dict): Dictionary of scenarios, each containing parameter modifications.
#     - its (int): Number of iterations to run the simulation.
#     - verbose (bool): Whether to display a progress bar during simulation.
#     - plot (bool): Whether to plot the average results for each scenario.

#     Returns:
#     - None
#     """
#     for scenario_name in scenarios:
#         for param in scenarios[scenario_name]:
#             params[param] = scenarios[scenario_name][param]
#         output = run_exec(params, its, verbose)
#         if plot:
#             plot_avg(output, title_add=f"scenarios/{scenario_name}")
#     return

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
            # exp_SWB = extract_data(output, 2)
            # exp_SWB_data[i, j] = np.mean(exp_SWB, axis=1)
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

    np.savez(f'data/stochplot_{param}.npz', SWB_data, exp_SWB_data, fin_data, exp_fin_data, nonfin_data, exp_nonfin_data, RFC_data, exp_RFC_data, soccap_data, exp_soccap_data)

    plot_stoch_param(SWB_data, param, param_samples, intervention_timesteps, int_var, title_add=title_add)
    if plot_components:
        plot_stoch_components(fin_data, exp_fin_data, nonfin_data, exp_nonfin_data, RFC_data, exp_RFC_data, soccap_data, exp_soccap_data, param, param_samples, intervention_timesteps, int_var, title_add=title_add)
    return

def run_two_var_heatmap(constants, runs, its, samples, params, bounds, title_add = "", hist_gap_comb=False):

    param_int_list = ["N", "hist_len", "intervention_gap"]
    init_fin, init_nonfin, init_SWB = init_ind_params(constants)
    param_1_values = 2**np.linspace(bounds[0][0], bounds[0][1], samples[0]) * 0.5
    param_2_values = 2**np.linspace(bounds[1][0], bounds[1][1], samples[1]) * 0.5
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
        title_add += "_hist_comb"
    np.save(f"data/heatmap_SWB_data{title_add}", SWB_data)
    np.save(f"data/heatmap_SWB_baseline{title_add}", SWB_baseline)
    np.save(f"data/heatmap_chg_data{title_add}", chg_data)
    two_var_heatmap(SWB_data, SWB_baseline, params, 16 / param_1_values, 16 / param_2_values, title_add=f"{title_add}")
    two_var_heatmap(chg_data, SWB_baseline, params, 16 / param_1_values, 16 / param_2_values, title_add=f"_chg{title_add}", per_person=True)
    return

def run_over_model(constants, runs, its, visualisations=[], networks=["SDA"], title_add=""):
    init_fin, init_nonfin, init_SWB = init_ind_params(constants)
    relevant_column_ids = [1, 8, 10, 14, 16, 18]

    intervention_timesteps = constants["int_ts"]
    int_var = constants["int_var"]
    for network in networks:
        print(f"---------{network}---------")
        constants["type"] = network
        data = np.empty((len(relevant_column_ids), runs, its+1, constants["N"]))
        for i in range(runs):
            output = run_exec(constants, its, init_fin, init_nonfin, init_SWB)
            for j, rel_id in enumerate(relevant_column_ids):
                data[j][i] = extract_data(output, rel_id)
        np.save(f"data/{network}_output_{runs}_{title_add}", data)

        # Visualise data
        if "plot_agent" in visualisations:
            agent_means = np.mean(data[:1, :, :, 0], axis=1)
            agent_stds = np.std(data[:1, :, :, 0], axis=1)
            plot_agent(agent_means, agent_stds, title_add= "_" + network + title_add)

        if "plot_avg" in visualisations:
            mean_over_agents = np.mean(data[:1], axis=3)
            means = np.mean(mean_over_agents, axis=1)
            stds = np.std(mean_over_agents, axis=1)
            plot_avg(means, stds, title_add= "_" + network + title_add)

        if "var_plot" in visualisations:
            SWB_var = np.var(data[0], axis=2)
            plot_var(SWB_var, intervention_timesteps, int_var, title_add= "_" + network + title_add)

        if "hist" in visualisations:
            begin_data = np.mean(data[0, :, 0], axis=0)
            end_data = np.mean(data[0, :, its], axis=0)
            hist_plot(begin_data, title_add="_begin" + "_" + network + title_add)
            hist_plot(end_data, title_add="_end" + "_" + network + title_add)

        if "degree" in visualisations:
            network_type = constants["type"]
            init_fin, init_nonfin, init_SWB = init_ind_params(constants)
            degrees = data[5, :, constants["burn_in_period"]:]
            SWB = data[0, :, constants["burn_in_period"]:]
            soccap = data[4, :, constants["burn_in_period"]:]
            degrees = degrees.flatten().reshape(-1, 1)
            SWB = SWB.flatten().reshape(-1, 1)
            soccap = soccap.flatten().reshape(-1, 1)
            degree_SWB_combo = np.hstack((degrees, SWB))
            degree_soccap_combo = np.hstack((degrees, soccap))
            df_SWB = pd.DataFrame(degree_SWB_combo, columns=["degree", "SWB"])
            df_SWB.to_csv(f"data/degree_SWB_{network_type}{title_add}.csv")
            df_soccap = pd.DataFrame(degree_soccap_combo, columns=["degree", "soccap"])
            df_soccap.to_csv(f"data/degree_soccap_{network_type}{title_add}.csv")
    return

def run_resilience(constants, runs, its, param, bounds, samples, int_ts, int_factors, int_var, title_add=""):
    amount_ints = len(int_ts)
    new_constants = constants.copy()
    new_constants["int_ts"] = int_ts
    new_constants["int_size"] = int_factors
    new_constants["int_var"] = [int_var] * amount_ints
    burn_in_period = new_constants["burn_in_period"]

    param_samples = np.linspace(bounds[0], bounds[1], samples)

    init_fin, init_nonfin, init_SWB = init_ind_params(new_constants)

    data = np.empty((samples, runs, len(int_ts), 2))
    for sample_index, sample in enumerate(param_samples):
        print(f"---{param}---{sample_index}---{int_var}---")
        new_constants[param] = sample
        for run in range(runs):
            output = run_exec(new_constants, its, init_SWB, init_fin, init_nonfin)
            SWB = extract_data(output, 1)
            mean_SWBs = np.mean(SWB, axis=1)
            equilibrium_SWB = np.mean(SWB[burn_in_period-20:burn_in_period])
            equilibrium_low_bound = equilibrium_SWB - np.std(mean_SWBs[:burn_in_period])
            int_ts = np.array(int_ts)
            for ts_index, ts in enumerate(int_ts):
                data[sample_index, run, ts_index, 0] = mean_SWBs[ts] - mean_SWBs[ts-1]
                for i in range(1, 60):
                    if mean_SWBs[ts-1+i] >= equilibrium_low_bound:
                        data[sample_index, run, ts_index, 1] = i
                        break

    np.save(f"data/resilience_{param}{title_add}", data)

def run_degree_SWB(constants, runs, its, title_add=""):
    network_type = constants["type"]
    init_fin, init_nonfin, init_SWB = init_ind_params(constants)
    degree_SWB_combo_list = []
    for run in range(runs):
        output = run_exec(constants, its, init_SWB, init_fin, init_nonfin)
        degrees = extract_data(output, 18)[constants["burn_in_period"]:].flatten().reshape(-1, 1)
        SWB = extract_data(output, 1)[constants["burn_in_period"]:].flatten().reshape(-1, 1)
        combo = np.hstack((degrees, SWB))
        degree_SWB_combo_list.append(combo)
    degree_SWB_combo = np.vstack(degree_SWB_combo_list)
    df = pd.DataFrame(degree_SWB_combo, columns=["degree", "SWB"])
    df.to_csv(f"data/degree_SWB_{network_type}{title_add}.csv")