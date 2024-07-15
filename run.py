"""
run file
"""
from run_functions import all_scenarios, stoch_plot_param, run_exec, run_two_var_heatmap, run_over_model, run_resilience, run_degree_SWB
from sa import param_space_behaviour, GSA
from visualisation import visualise, plot_avg, plot_agent, SWB_gif
from parameters import params
from scenarios import scenarios
from stochs import param_dict
import numpy as np

ITERATIONS = 250
RUNS = 1
SAMPLES = 3
edit_params = ["segregation", "beta", "rec_intervention_factor", "intervention_gap", "hist_len", "fb_fin", "dummy"]
bounds = [(0, 10), (0, 20), (0.5, 2), (1, 50), (1, 50), (0, 2), (0, 5)]


# run_degree_SWB(params, RUNS, ITERATIONS)

# # Run the model once
output = run_exec(params, ITERATIONS)

# # Plot simple plots
# plot_agent(output, title_add="_BA")
# plot_avg(output, title_add="_BA")


# # Gathering data of the system
# if __name__ == '__main__':
#     param_space_behaviour(params, ITERATIONS, 4096 , edit_params, bounds)

# # One-Factor-At-A-Time 
# # 
# run_over_model(params, RUNS, ITERATIONS, networks=["SDA"], visualisations=["plot_agent", "plot_avg"], title_add="test")

# Stochastic plot for 1 parameter with bounds and param samples
# for param in param_dict:
#     print(f"------{param}------")
#     params_copy = params.copy()
#     stoch_plot_param(params_copy, RUNS, ITERATIONS, param, param_dict[param], SAMPLES, plot_components=True, title_add="OFAT/")

# # Run all scenarios and plot average SWB
# all_scenarios(params, scenarios, ITERATIONS)

# # SWB gif of network
# visualise(model, output)

# # Plot histogram of SWB over time
# SWB_gif(model, output, ITERATIONS, fps=5, name="SWB_hist", xlabel="SWB score",
#         ylabel="Amount", xlim=(0, 10), ylim=[0, 10])

# # Resilience tests
# run_resilience(params, RUNS, ITERATIONS, "soc_cap_base", [0.01, 1], SAMPLES, [60, 120, 180, 240], [5/6, 4/6, 3/6, 2/6], "fin", title_add="fin")
# run_resilience(params, RUNS, ITERATIONS, "soc_cap_inf", [0.25, 4], SAMPLES, [60, 120, 180, 240], [5/6, 4/6, 3/6, 2/6], "fin", title_add="fin")

# run_resilience(params, RUNS, ITERATIONS, "soc_cap_base", [0.01, 1], SAMPLES, [60, 120, 180, 240], [5/6, 4/6, 3/6, 2/6], "nonfin", title_add="nonfin")
# run_resilience(params, RUNS, ITERATIONS, "soc_cap_inf", [0.25, 4], SAMPLES, [60, 120, 180, 240], [5/6, 4/6, 3/6, 2/6], "nonfin", title_add="nonfin")

# # Heatmap
# run_two_var_heatmap(params, RUNS, ITERATIONS, (3, 3), ["intervention_gap", "hist_len"], [(1, 7), (1, 7)], title_add="hist_int_testtest", hist_gap_comb=False)

# # Specific run functions
# # Sensitivity Analysis
# if __name__ == '__main__':
#     sa = GSA(params, ITERATIONS, 1024, edit_params, bounds, sa_type="Pawn")
#     print("---------------")
#     print(sa)
