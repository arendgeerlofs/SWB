"""
run file
"""
from run_functions import *
from sa import param_space_behaviour
from visualisation import plot_avg, plot_agent, SWB_gif
from parameters import params
from stochs import param_dict

ITERATIONS = 250
RUNS = 50
SAMPLES = 6
edit_params = ["segregation", "beta", "rec_intervention_factor", "intervention_gap",
                "hist_len", "fb_fin", "dummy"]
bounds = [(0, 10), (0, 20), (0.5, 2), (1, 50), (1, 50), (0, 2), (0, 5)]


# Create degree SWB pairs and plot their relation
run_degree_SWB(params, RUNS, ITERATIONS)

# One-Factor-At-A-Time
run_over_model(params, RUNS, ITERATIONS, networks=["SDA"],
               visualisations=["plot_agent", "plot_avg"], title_add="")

# Stochastic plot for 1 parameter with bounds and param samples
for param in param_dict:
    print(f"------{param}------")
    params_copy = params.copy()
    stoch_plot_param(params_copy, RUNS, ITERATIONS, param, param_dict[param], SAMPLES,
                     plot_components=True, title_add="")

# Create example model
model = init_model(params)

# Run the model once
output = run_exec(params, ITERATIONS)

# Plot histogram of SWB over time
SWB_gif(model, output, ITERATIONS, fps=5, name="SWB_hist", xlabel="SWB score",
        ylabel="Amount")

# Resilience tests
run_resilience(params, RUNS, ITERATIONS, "soc_cap_base", [0.01, 1], SAMPLES,
               [125, 150, 175, 200], [5/6, 4/6, 3/6, 2/6], "fin", title_add="fin_base")
run_resilience(params, RUNS, ITERATIONS, "soc_cap_inf", [0.25, 4], SAMPLES,
               [125, 150, 175, 200], [5/6, 4/6, 3/6, 2/6], "fin", title_add="fin_inf")

run_resilience(params, RUNS, ITERATIONS, "soc_cap_base", [0.01, 1], SAMPLES,
               [125, 150, 175, 200], [5/6, 4/6, 3/6, 2/6], "nonfin", title_add="nonfin_base")
run_resilience(params, RUNS, ITERATIONS, "soc_cap_inf", [0.25, 4], SAMPLES,
               [125, 150, 175, 200], [5/6, 4/6, 3/6, 2/6], "nonfin", title_add="nonfin_inf")

# Heatmap
run_two_var_heatmap(params, RUNS, ITERATIONS, (3, 3), ["intervention_gap", "hist_len"],
                    [(1, 7), (1, 7)], title_add="hist_int", hist_gap_comb=False)

# Should not be run at the same time as the other functions as it results in problems
# With the multiprocessing
# Gathering data of the system
if __name__ == '__main__':
    param_space_behaviour(params, ITERATIONS, 2, edit_params, bounds)
