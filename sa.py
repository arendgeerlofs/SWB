import numpy as np
from tqdm import tqdm
from SALib.analyze.sobol import analyze
from SALib.analyze import pawn
from SALib.sample.sobol import sample
from functions import extract_data
from model import init_model
import dask


def GSA(constants, its, samples, parameters=[], bounds=[[]], sa_type = "Normal"):
    """
    Global Sensitivity Analysis
    """

    # Define the model inputs
    problem = {
        'num_vars': len(parameters),
        'names': parameters,
        'bounds': bounds
    }

    # Generate samples
    param_values = sample(problem, samples)
    
    data = np.empty(len(param_values))
    for index, params in tqdm(enumerate(param_values)):
        for param_ind, param in enumerate(params):
            constants[parameters[param_ind]] = param
        model = init_model(constants)
        output = model.simulate(its, show_tqdm=False)
        SWB = extract_data(constants["N"], output, 1)
        data[index] = np.mean(SWB[-1])
    
    # dask.compute(data)
    # Perform analysis
    if sa_type == "Normal":
        Si = analyze(problem, data, print_to_console=True)
    elif sa_type == "Pawn":
        Si = pawn.analyze(problem, param_values, data, S=10, print_to_console=False)

    return Si

def LSA(constants, its, samples, parameters=[], bounds=[[]]):
    """"
    Local Sensitivity analysis using OFAT
    """
    data = np.array((len(parameters), samples))
    for index, param in enumerate(parameters):
        param_values = np.linspace(bounds[index][0], bounds[index][1], samples)
        for i, value in enumerate(param_values):
            constants[param] = value
            model = init_model(constants)
            output = model.simulate(its)
            SWB = extract_data(constants["N"], output, 1)
            data[index][i] = np.mean(SWB[-1])

    return data