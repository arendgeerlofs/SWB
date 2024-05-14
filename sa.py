import numpy as np
from tqdm import tqdm
import multiprocessing
from SALib.analyze import sobol
from SALib.analyze.sobol import analyze
from SALib.analyze import pawn
from SALib.sample.sobol import sample
from SALib.sample import saltelli
from functions import extract_data
from run_functions import run_exec

def run_model(param_values, constants, problem, its, inits, output_queue):
    """
    Function to run the model with given parameter values
    """
    data = np.empty(len(param_values))
    init_fin, init_nonfin, init_SWB = inits
    for index, params in enumerate(param_values):
        new_constants = constants.copy()
        for param_ind, param in enumerate(params):
            new_constants[problem['names'][param_ind]] = param
        output = run_exec(new_constants, its, init_fin, init_nonfin, init_SWB, verbose=False)
        SWB = extract_data(new_constants["N"], output, 1)
        data[index] = np.mean(SWB[-1])
    output_queue.put(data)

def GSA(constants, its, samples, parameters=[], bounds=[[]], sa_type="Normal"):
    """
    Global Sensitivity Analysis
    """

    # Define the model inputs
    problem = {
        'num_vars': len(parameters),
        'names': parameters,
        'bounds': bounds
    }

    # Create initial values
    init_fin = np.random.uniform(constants["L_low"], constants["L_high"], constants['N'])
    init_nonfin = np.random.uniform(constants["L_low"], constants["L_high"], constants['N'])
    init_SWB = np.clip(np.random.normal(constants["SWB_mu"], constants["SWB_sd"], constants['N']), 0, 10)
    inits = (init_fin, init_nonfin, init_SWB)

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
        process = multiprocessing.Process(target=run_model, args=(param_chunk, constants, problem, its, inits, output_queue))
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

def LSA(constants, its, samples, parameters=[], bounds=[[]]):
    """"
    Local Sensitivity analysis using OFAT
    """
    data = np.array((len(parameters), samples))
    for index, param in enumerate(parameters):
        param_values = np.linspace(bounds[index][0], bounds[index][1], samples)
        new_constants = constants
        for i, value in enumerate(param_values):
            new_constants[param] = value
            output = exec(new_constants, its, verbose=False)
            SWB = extract_data(new_constants["N"], output, 1)
            data[index][i] = np.mean(SWB[-1])

    return data