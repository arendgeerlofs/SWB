import numpy as np
from tqdm import tqdm
from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample
from functions import extract_data
from model import init_model


def GSA(constants, its, samples, parameters=[], bounds=[[]]):
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
        print(params)
        for param_ind, param in enumerate(params):
            constants[parameters[param_ind]] = param
        model = init_model(constants)
        output = model.simulate(its, show_tqdm=False)
        SWB = extract_data(constants["N"], output, 1)
        data[index] = np.mean(SWB[-1])
    
    # Perform analysis
    Si = analyze(problem, data, print_to_console=True)

    # Print the first-order sensitivity indices
    print(Si['S1'])

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