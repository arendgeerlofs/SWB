"""
run file
"""
from run_functions import all_scenarios, stoch_plot_param
from sa import GSA, LSA
from visualisation import visualise, plot_avg, plot_for_one, SWB_gif
from parameters import params
from scenarios import scenarios

ITERATIONS = 100
RUNS = 10

# model = init_model(params)
# output = run_model(model, ITERATIONS)

# plot_for_one(output)
# plot_avg(output)

# stoch_plot_param(params, RUNS, ITERATIONS, "event_size", [0, 1], 10)

# all_scenarios(params, scenarios, ITERATIONS)
# SWB_gif(model, output, ITERATIONS, fps=5, name="SWB_hist", xlabel="SWB score",
#         ylabel="Amount", xlim=(0, 10), ylim=[0, 10])
# visualise(model, output)
if __name__ == '__main__':
    edit_params = ["fin_event_prob", "event_size", "intervention_size", "intervention_gap", "hist_len", "SWB_mu", "SWB_sd", "soc_w", "fb_fin", "fb_nonfin",]
    bounds = [(0, 0.5), (0, 4), (0, 5), (1, 12), (1, 20), (0, 10), (0, 5), (0, 1), (0, 0.5), (0, 0.5)]
    sa = GSA(params, ITERATIONS, 1024, edit_params, bounds, sa_type="Pawn")
    print("---------------")
    print(sa)

