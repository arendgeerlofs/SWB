"""
run file
"""
from run_functions import all_scenarios, stoch_plot_param, run_exec
from sa import GSA, LSA
from visualisation import visualise, plot_avg, plot_for_one, SWB_gif
from parameters import params
from scenarios import scenarios

ITERATIONS = 100
RUNS = 10

# output = run_exec(params, ITERATIONS)
# plot_for_one(output)
# plot_avg(output)

# stoch_plot_param(params, RUNS, ITERATIONS, "event_size", [0, 1], 10)

# all_scenarios(params, scenarios, ITERATIONS)
# SWB_gif(model, output, ITERATIONS, fps=5, name="SWB_hist", xlabel="SWB score",
#         ylabel="Amount", xlim=(0, 10), ylim=[0, 10])
# visualise(model, output)
if __name__ == '__main__':
    edit_params = ["fin_event_prob", "nonfin_event_prob", "event_size", "rec_intervention_size", "intervention_gap", "hist_len", "fb_fin", "fb_nonfin", "soc_cap_base", "soc_cap_inf", "dummy"]
    bounds = [(0, 0.5), (0, 0.5), (0, 4), (0, 5), (1, 12), (1, 20), (0, 0.5), (0, 0.5), (0.01, 0.05), (0.25, 4), (0, 10)]
    sa = GSA(params, ITERATIONS, 32, edit_params, bounds, sa_type="Normal")
    print("---------------")
    print(sa)

