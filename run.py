"""
run file
"""

from DyNSimF import init_model, run_model, visualise

ITERATIONS = 50
RUNS = 10

model = init_model()
output = run_model(model, ITERATIONS)
#visualise(model, output)
