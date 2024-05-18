"""
run file
"""
from run_functions import all_scenarios, stoch_plot_param, run_exec
from sa import GSA, LSA
from visualisation import visualise, plot_avg, plot_for_one, SWB_gif
from parameters import params
from scenarios import scenarios
from stochs import param_dict

ITERATIONS = 200
RUNS = 25
SAMPLES = 5

# One run of the model
output = run_exec(params, ITERATIONS)

# Plot simple plots
plot_for_one(output)
plot_avg(output)

# Stochastic plot for 1 parameter with bounds and param samples
# for param in param_dict:
#     print(f"------{param}------")
#     params_copy = params.copy()
#     stoch_plot_param(params_copy, RUNS, ITERATIONS, param, param_dict[param], SAMPLES)


# Run all scenarios and plot average SWB
# all_scenarios(params, scenarios, ITERATIONS)

# Plot histogram of SWB over time
# SWB_gif(model, output, ITERATIONS, fps=5, name="SWB_hist", xlabel="SWB score",
#         ylabel="Amount", xlim=(0, 10), ylim=[0, 10])

# SWB gif of network
# visualise(model, output)

# Sensitivity Analysis
# if __name__ == '__main__':
#     edit_params = ["fin_event_prob", "nonfin_event_prob", "event_size", "rec_intervention_size", "intervention_gap", "hist_len", "fb_fin", "fb_nonfin", "soc_cap_base", "soc_cap_inf", "dummy"]
#     bounds = [(0, 0.5), (0, 0.5), (0, 4), (0, 5), (1, 12), (1, 20), (0, 0.5), (0, 0.5), (0.01, 0.05), (0.25, 4), (0, 10)]
#     sa = GSA(params, ITERATIONS, 1024, edit_params, bounds, sa_type="Pawn")
#     print("---------------")
#     print(sa)

