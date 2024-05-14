
from model import init_model, run_model
from visualisation import plot_avg
from functions import extract_data
import numpy as np


def run_exec(constants, iterations, init_fin=False, init_nonfin=False, init_SWB=False, verbose=True):
    model = init_model(constants, init_fin, init_nonfin, init_SWB)
    output = run_model(model, iterations, verbose)
    return output

def all_scenarios(params, scenarios, its, verbose=True, plot=True):
    for scenario_name in scenarios:
        for param_id, param in enumerate(scenarios[scenario_name]):
            params[param] = scenarios[scenario_name][param]
        output = exec(params, its, verbose)
        if plot:
            plot_avg(output, name_addition=f"scenarios/{scenario_name}")
    return

def stoch_plot_param(constants, runs, its, param, bounds, steps):
    SWB_data = np.empty((steps, runs, its+1))
    for i, param_value in enumerate(np.linspace(bounds[0], bounds[1], steps)):
        constants[param] = param_value
        for j in range(runs):
            output = run_exec(constants, its)
            SWB = extract_data(constants["N"], output, 1)
            SWB_data[i, j] = np.mean(SWB, axis=1)
    print(SWB_data)
    return
        