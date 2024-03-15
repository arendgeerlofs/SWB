"""
run file
"""

from DyNSimF import init_model, run_model
from visualisation import visualise, plot, SWB_gif

ITERATIONS = 100
RUNS = 10

model = init_model()
output = run_model(model, ITERATIONS)
plot(output)
SWB_gif(output, ITERATIONS, fps=5, name="SWB_hist", xlabel="SWB score", 
        ylabel="Amount", xlim=[0, 10], ylim=[0, 10])
# visualise(model, output)
