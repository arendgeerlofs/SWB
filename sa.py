import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from SALib.analyze import sobol, pawn
from SALib.sample.sobol import sample
from functions import extract_data, init_ind_params, get_all_data
from run_functions import run_exec


def run_model(param_values, constants, problem, its, inits, output_queue, output_type="GSA"):
    """
    Run the model with given parameter values and collect results.

    Parameters:
    - param_values (ndarray): Array of parameter values for the simulation runs.
    - constants (dict): Dictionary of model constants.
    - problem (dict): Dictionary defining the problem for sensitivity analysis.
    - its (int): Number of iterations to run the model.
    - inits (tuple): Initial values for financial, non-financial, and SWB states.
    - output_queue (multiprocessing.Queue): Queue to store results from the model runs.

    This function updates the model constants with new parameter values, runs the model, 
    extracts the SWB data, calculates the mean SWB at the last timestep, and stores the results in the output queue.
    """
    param_int_list = ["N", "hist_len", "intervention_gap"]
    if output_type == "GSA":
        data = np.empty(len(param_values))
    elif output_type == "All":
        print(np.shape(param_values)[1])
        data = np.empty((len(param_values), np.shape(param_values)[1]+(18+2)*2+5))
    
    init_fin, init_nonfin, init_SWB = inits

    for index, params in enumerate(param_values):
        new_constants = constants.copy()
        for param_ind, param in enumerate(params):
            if problem['names'][param_ind] in param_int_list:
                param = int(param)
            new_constants[problem['names'][param_ind]] = param
        output = run_exec(new_constants, its, init_fin, init_nonfin, init_SWB, verbose=False)

        if output_type == "GSA":
            SWB = extract_data( output, 1)
            data[index] = np.mean(SWB[:-50])
        elif output_type == "All":
            output_data = get_all_data(output, new_constants)
            param_array = params
            data[index] = np.concatenate((param_array, output_data))
    output_queue.put(data)

def GSA(constants, its, samples, parameters=[], bounds=[[]], sa_type="Normal"):
    """
    Global Sensitivity Analysis (GSA) using Sobol or Pawn methods.

    Parameters:
    - constants (dict): Dictionary of model constants.
    - its (int): Number of iterations to run the model.
    - samples (int): Number of samples for the sensitivity analysis.
    - parameters (list): List of parameter names to be analyzed.
    - bounds (list): List of bounds for each parameter.
    - sa_type (str): Type of sensitivity analysis ('Normal' for Sobol, 'Pawn' for PAWN).

    Returns:
    - Si (dict): Sensitivity indices calculated by the chosen sensitivity analysis method.

    This function defines the problem, initializes parameters, generates samples,
    runs the model using multiprocessing, and performs the chosen sensitivity analysis.
    """
    # Define the model inputs
    problem = {
        'num_vars': len(parameters),
        'names': parameters,
        'bounds': bounds
    }

    # Create initial values
    inits = init_ind_params(constants)

    # Generate samples
    param_values = sample(problem, samples)

    # Split param_values into chunks for multiprocessing
    chunk_size = len(param_values) // multiprocessing.cpu_count()
    param_chunks = [param_values[i:i + chunk_size] for i in range(0, len(param_values), chunk_size)]

    # Create a multiprocessing Queue to collect results from worker processes
    output_queue = multiprocessing.Queue()

    # Create and start worker processes
    processes = []
    for param_chunk in param_chunks:
        process = multiprocessing.Process(target=run_model, args=(param_chunk, constants, problem, its, inits, output_queue, "GSA"))
        processes.append(process)
        process.start()

    # Collect results from worker processes
    results = []
    for _ in range(len(param_chunks)):
        results.append(output_queue.get())

    # Join worker processes
    for process in processes:
        process.join()

    # Combine results from all processes
    data = np.concatenate(results)

    # Perform analysis
    if sa_type == "Normal":
        Si = sobol.analyze(problem, data, print_to_console=True)
    elif sa_type == "Pawn":
        Si = pawn.analyze(problem, param_values, data, S=10, print_to_console=False)

    return Si

def param_space_behaviour(constants, its, samples, parameters=[], bounds=[[]]):
    """
    Global Sensitivity Analysis (GSA) using Sobol or Pawn methods.

    Parameters:
    - constants (dict): Dictionary of model constants.
    - its (int): Number of iterations to run the model.
    - samples (int): Number of samples for the sensitivity analysis.
    - parameters (list): List of parameter names to be analyzed.
    - bounds (list): List of bounds for each parameter.
    - sa_type (str): Type of sensitivity analysis ('Normal' for Sobol, 'Pawn' for PAWN).

    Returns:
    - Si (dict): Sensitivity indices calculated by the chosen sensitivity analysis method.

    This function defines the problem, initializes parameters, generates samples,
    runs the model using multiprocessing, and performs the chosen sensitivity analysis.
    """
    # Define the model inputs
    problem = {
        'num_vars': len(parameters),
        'names': parameters,
        'bounds': bounds
    }

    # Create initial values
    inits = init_ind_params(constants)

    # Generate samples
    param_values = sample(problem, samples)

    # Split param_values into chunks for multiprocessing
    chunk_size = len(param_values) // multiprocessing.cpu_count()
    param_chunks = [param_values[i:i + chunk_size] for i in range(0, len(param_values), chunk_size)]

    # Create a multiprocessing Queue to collect results from worker processes
    output_queue = multiprocessing.Queue()

    # Create and start worker processes
    processes = []
    for param_chunk in param_chunks:
        process = multiprocessing.Process(target=run_model, args=(param_chunk, constants, problem, its, inits, output_queue, "All"))
        processes.append(process)
        process.start()

    # Collect results from worker processes
    results = []
    for _ in range(len(param_chunks)):
        results.append(output_queue.get())

    # Join worker processes
    for process in processes:
        process.join()

    # Combine results from all processes
    data = np.vstack(results)

    # Set columns names
    columns = parameters + ["Mean SWB norm", "Var SWB norm", "Mean SWB", "Var SWB", "Mean SWB exp", "Var SWB exp", 
                            "Mean SWB community", "Var SWB_community", "Mean habituation",
                            "Var habituation", "Mean sensitisation", "Var sensitisation", "Mean desensitisation", "Var desensitisation",
                            "Mean social w", "Var social w", "Mean financial", "Var financial", "Mean financial exp", "Var financial exp", 
                            "Mean non-financial", "Var non-financial", "Mean non-financial exp", "Var non-financial exp", 
                            "Mean financial sens", "Var financial sens", "Mean non-financial sens", "Var non-financial sens", 
                            "Mean RFC", "Var RFC", "Mean RFC exp", "Var RFC exp", "Mean social capital", "Var social capital", 
                            "Mean social capital exp", "Var social capital exp", " Mean SWB before intervention", "Var SWB before intervention", 
                            "Mean SWB after intervention", " Var SWB after intervention", "System behaviour", "Percent changed positive", 
                            "Percent changed negative", "Percent not changed", "System changed"]
    # Perform analysis
    df = pd.DataFrame(data, columns=columns)
    df.to_csv("data/param_space_behaviour_results.csv")
    return data[40]

def LSA(constants, its, samples, parameters=[], bounds=[[]]):
    """
    Local Sensitivity Analysis (LSA) using One-Factor-At-a-Time (OFAT) method.

    Parameters:
    - constants (dict): Dictionary of model constants.
    - its (int): Number of iterations to run the model.
    - samples (int): Number of samples for the sensitivity analysis.
    - parameters (list): List of parameter names to be analyzed.
    - bounds (list): List of bounds for each parameter.

    Returns:
    - data (ndarray): Array of results from the sensitivity analysis.

    This function performs local sensitivity analysis by varying one parameter at a time
    within the specified bounds, running the model, and calculating the mean SWB at the last timestep.
    """
    data = np.array((len(parameters), samples))
    for index, param in enumerate(parameters):
        param_values = np.linspace(bounds[index][0], bounds[index][1], samples)
        new_constants = constants
        for i, value in enumerate(param_values):
            new_constants[param] = value
            output = exec(new_constants, its, verbose=False)
            SWB = extract_data(output, 1)
            data[index][i] = np.mean(SWB[-1])

    return data
