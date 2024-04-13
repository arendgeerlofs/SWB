"""
run file
"""
from model import init_model, run_model
from sa import GSA, LSA
from visualisation import visualise, plot, plot_for_one, SWB_gif
from parameters import constants

ITERATIONS = 50
# RUNS = 10

model = init_model(constants)
output = run_model(model, ITERATIONS)

plot_for_one(output, plot_from=0)
plot(output)
# SWB_gif(output, ITERATIONS, fps=5, name="SWB_hist", xlabel="SWB score",
#         ylabel="Amount", xlim=(0, 10), ylim=[0, 10])
# visualise(model, output)

# params = ["event_prob", "event_size", "intervention_size", "intervention_gap", "hist_len", "SWB_mu", "SWB_sd", "soc_w"]
# bounds = [(0, 0.5), (0, 10), (0, 5), (1, 12), (1, 10), (0, 10), (0, 5), (0, 1)]
# sa = GSA(constants, ITERATIONS, 256, params, bounds, sa_type="Pawn")
# print("---------------")
# print(sa)