"""
run file
"""

from model import init_model, run_model
from sa import GSA, LSA
from visualisation import visualise, plot, SWB_gif
from parameters import constants

ITERATIONS = 100
RUNS = 10

# model = init_model(constants)
# output = run_model(model, ITERATIONS)

# plot(output)
# SWB_gif(output, ITERATIONS, fps=5, name="SWB_hist", xlabel="SWB score", 
#         ylabel="Amount", xlim=(0, 10), ylim=[0, 10])
# visualise(model, output)

params = ["event_prob", "event_size", "intervention_size", "intervention_gap"]
bounds = [(0, 0.5), (0, 10), (0, 5), (1, 12)]
sa = GSA(constants, ITERATIONS, 512, params, bounds)
print("---------------")
print(sa)