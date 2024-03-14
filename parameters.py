"""
File containing all the parameter values and constants
"""
network_parameters = {
    # Network parameters
    "type" : "Rd",
    'm' : 1,
    'p' : 0.1,
    'segregation': 0.9,
    }

constants = {
    # Population size
    'N' : 1000,
    # Likert scale low and high
    "L_low" : 0,
    "L_high" : 10,
    # Max value for SWB
    "SWB_high" : 10,
    # Event size
    "event_prob": 0.02,
    "event_size" : 1,
    # Parameter indicating change of model during simulations
    "upd_net" : False
}
