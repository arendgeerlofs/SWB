"""
File containing all the parameter values and constants
"""

constants = {
    # Network parameters
    "type" : "Rd",
    'N' : 100,
    'm' : 1,
    'p' : 0.1,
    'segregation': 0.9,
    # Likert scale low and high
    "L_low" : 1,
    "L_high" : 10,
    # SWB initialisation
    "SWB_mu" : 7,
    "SWB_sd" : 2,
    # Event size
    "event_prob": 0.05,
    "event_size" : 1,
    # Parameter indicating change of model during simulations
    "upd_net" : False,
    # Intervention size
    "intervention_size" : 1,
    "intervention_gap" : 10,
    # History length
    "hist_len" : 10,
    # Social comparison weight
    "soc_w" : 0.5,
}