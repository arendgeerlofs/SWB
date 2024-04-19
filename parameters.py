"""
File containing all the parameter values and constants
"""

params = {
    # Network parameters
    "type" : "SDA",
    'N' : 100,
    'm' : 1,
    'p' : 0.1,
    'segregation': 0.9,
    'beta' : 0.1,
    # Likert scale low and high
    "L_low" : 1,
    "L_high" : 10,
    # SWB initialisation
    "SWB_mu" : 7,
    "SWB_sd" : 2,
    # Event size
    "fin_event_prob": 0.00,
    "nonfin_event_prob": 0.00,
    "event_size" : 1,
    # Parameter indicating change of model during simulations
    "upd_net" : True,
    # Pulse intervention size
    "rec_intervention_size" : 0,
    "intervention_gap" : 10,
    # Set intervention
    "int_ts": [10, 10],
    "int_size": [1, 1],
    "int_var": ["fin", "nonfin"],
    # History length
    "hist_len" : 10,
    # Feedback parameters
    "fb_fin": 0.0,
    "fb_nonfin": 0.0,
}