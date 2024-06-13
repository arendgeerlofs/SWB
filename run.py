"""
run file
"""
# from run_functions import all_scenarios, stoch_plot_param, run_exec, run_var_plot, run_two_var_heatmap
from sa import param_space_behaviour#, GSA
# from visualisation import visualise, plot_avg, plot_for_one, SWB_gif
from parameters import params
# from scenarios import scenarios
# from stochs import param_dict

ITERATIONS = 200
RUNS = 1
SAMPLES = 1

# One run of the model
# output = run_exec(params, ITERATIONS)

# Plot simple plots
# plot_for_one(output)
# plot_avg(output)

# # Stochastic plot for 1 parameter with bounds and param samples
# for param in param_dict:
#     print(f"------{param}------")
#     params_copy = params.copy()
#     stoch_plot_param(params_copy, RUNS, ITERATIONS, param, param_dict[param], SAMPLES, plot_components=True, title_add="no_rec/")


# # Var plot
# run_var_plot(params, RUNS, ITERATIONS)

# Run all scenarios and plot average SWB
# all_scenarios(params, scenarios, ITERATIONS)

# Plot histogram of SWB over time
# SWB_gif(model, output, ITERATIONS, fps=5, name="SWB_hist", xlabel="SWB score",
#         ylabel="Amount", xlim=(0, 10), ylim=[0, 10])

# SWB gif of network
# visualise(model, output)

edit_params = ["segregation", "beta", "fin_event_prob", "nonfin_event_prob", "event_size", "net_upd_freq", "rec_intervention_factor", "intervention_gap", "hist_len", "fb_fin", "fb_nonfin", "soc_cap_base", "soc_cap_inf", "dummy"]
bounds = [(0, 2), (0, 20), (0, 1), (0, 1), (0, 5), (0, 1), (0.5, 2), (1, 50), (1, 50), (0, 2), (0, 2), (0.01, 1), (0.25, 4), (0, 10)]

# # Sensitivity Analysis
# if __name__ == '__main__':
#     sa = GSA(params, ITERATIONS, 1024, edit_params, bounds, sa_type="Pawn")
#     print("---------------")
#     print(sa)

# System data
if __name__ == '__main__':
    data = param_space_behaviour(params, ITERATIONS, 2, edit_params, bounds)
    # print("---------------")
    # print(data)

# Heatmap
# run_two_var_heatmap(params, RUNS, ITERATIONS, (7, 11), ["hist_comb", "rec_intervention_factor"], [(1, 7), (1, 2)], title_add="exp", hist_gap_comb=True)