"""
run file
"""

from DyNSimF import init_model, run_model, visualise, plot

ITERATIONS = 10
RUNS = 10

model = init_model()
output = run_model(model, ITERATIONS)
plot(output)
#visualise(model, output)
