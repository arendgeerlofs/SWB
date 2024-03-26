"""
File containing all the parameter values and constants
"""

constants = {
    # Population size
    'N' : 100,
    # Likert scale low and high
    "L_low" : 0,
    "L_high" : 10,
    # Max value for SWB
    "SWB_high" : 10,
    # Event size
    "event_prob": 0.05,
    "event_size" : 1,
    # Parameter indicating change of model during simulations
    "upd_net" : False,
    # Intervention size
    "intervention_size" : 1,
    "intervention_gap" : 10,
    # History length
    "hist_len" : 10
}

network_parameters = {
    # Network parameters
    "type" : "Rd",
    'N' : constants["N"],
    'm' : 1,
    'p' : 0.1,
    'segregation': 0.9,
    }
