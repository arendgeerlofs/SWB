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
    "fin_event_prob": 0.0,
    "nonfin_event_prob": 0.0,
    "event_size" : 1,
    # Parameter indicating change of model during simulations
    "upd_net" : False,
    # Pulse intervention size
    "rec_intervention_size" : 0,
    "intervention_gap" : 3,
    # Set intervention
    "int_ts": [10, 30, 50, 70],
    "int_size": [1, -1, 1, -1],
    "int_var": ["fin", "fin", "nonfin", "nonfin"],
    # History length
    "hist_len" : 10,
    # Feedback parameters
    "fb_fin": 0.1,
    "fb_nonfin": 0.3,
    # soc cap parameters
    "soc_cap_base": 0.03,
    "soc_cap_inf": 1,
    # Dummy variable
    "dummy": 5,
}
