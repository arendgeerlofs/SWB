# import numpy as np
# import multiprocessing
# from SALib.analyze import sobol
# from SALib.sample.sobol import sample
# from tqdm import tqdm
# from functions import extract_data
# from model import init_model, run
# from SALib.analyze import pawn
# from parameters import params

# def run_model(param_values, constants, problem, its, output_queue):
#     """
#     Function to run the model with given parameter values
#     """
#     data = np.empty(len(param_values))
#     for index, params in enumerate(param_values):
#         new_constants = constants.copy()
#         for param_ind, param in enumerate(params):
#             new_constants[problem['names'][param_ind]] = param
#         output = run(new_constants, its, verbose=False)
#         SWB = extract_data(new_constants["N"], output, 1)
#         data[index] = np.mean(SWB[-1])
#     output_queue.put(data)

# def GSA(constants, its, samples, parameters=[], bounds=[[]], sa_type="Normal"):
#     """
#     Global Sensitivity Analysis
#     """

#     # Define the model inputs
#     problem = {
#         'num_vars': len(parameters),
#         'names': parameters,
#         'bounds': bounds
#     }

#     # Generate samples
#     param_values = sample(problem, samples)

#     # Split param_values into chunks for multiprocessing
#     chunk_size = len(param_values) // multiprocessing.cpu_count()
#     param_chunks = [param_values[i:i + chunk_size] for i in range(0, len(param_values), chunk_size)]

#     # Create a multiprocessing Queue to collect results from worker processes
#     output_queue = multiprocessing.Queue()

#     # Create and start worker processes
#     processes = []
#     for param_chunk in param_chunks:
#         process = multiprocessing.Process(target=run_model, args=(param_chunk, constants, problem, its, output_queue))
#         processes.append(process)
#         process.start()

#     # Collect results from worker processes
#     results = []
#     for _ in range(len(param_chunks)):
#         results.append(output_queue.get())

#     # Join worker processes
#     for process in processes:
#         process.join()

#     # Combine results from all processes
#     data = np.concatenate(results)

#     # Perform analysis
#     if sa_type == "Normal":
#         Si = sobol.analyze(problem, data, print_to_console=True)
#     elif sa_type == "Pawn":
#         Si = pawn.analyze(problem, param_values, data, S=10, print_to_console=False)

#     return Si

# if __name__ == '__main__':
#     ITERATIONS = 10

#     edit_params = ["fin_event_prob", "event_size", "intervention_size", "intervention_gap", "hist_len", "SWB_mu", "SWB_sd", "soc_w"]
#     bounds = [(0, 0.5), (0, 10), (0, 5), (1, 12), (1, 10), (0, 10), (0, 5), (0, 1)]
#     sa = GSA(params, ITERATIONS, 256, edit_params, bounds)
#     print("---------------")
#     print(sa)
