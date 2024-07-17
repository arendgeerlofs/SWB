"""
File containing default parameter values
"""

params = {
    # Network parameters
    "type" : "SDA",  # Type of the network model.
    'N' : 100,  # Number of nodes in the network.
    'm' : 5,  # Initial number of edges to attach from a new node to existing nodes.
    'p' : 0.1,  # Probability of adding a new edge between two existing nodes.
    'segregation': 3,  # Segregation parameter influencing the likelihood of connection changes.
    'beta' : 1,  # Beta parameter influencing the connection probability.

    # Likert scale low and high
    "L_low" : 0,  # Lower bound of the Likert scale.
    "L_high" : 10,  # Upper bound of the Likert scale.

    # SWB initialization
    "SWB_mu" : 50,  # Mean of the initial SWB (Subjective Well-Being) distribution.
    "SWB_sd" : 1,  # Standard deviation of the initial SWB distribution.

    # Initial period to burn in the model
    "burn_in_period" : 100,  # Initial period without interventions to allow the model to stabilize.

    # Event size
    "fin_event_prob": 0.1,  # Probability of a financial event occurring.
    "nonfin_event_prob": 0.1,  # Probability of a non-financial event occurring.
    "event_size" : 0.1,  # Magnitude of the event size.

    # Parameter indicating change of model during simulations
    "upd_net" : True,  # Boolean to indicate if the network should be updated during simulations.
    "net_upd_freq" : 0.05, # Probability of network update occurring.

    # Pulse intervention size
    "rec_intervention_factor" : 1.1,  # Factor of the recurring intervention.
    "intervention_gap" : 1000,  # Gap between recurring interventions.

    # Set intervention
    "int_ts": [110, 130, 150, 170],  # List of timesteps at which interventions occur.
    "int_size": [11/10, 10/11, 11/10, 10/11],  # List of intervention factors.
    "int_var": ["fin", "fin", "nonfin", "nonfin"],  # List of intervention variables.

    # History length
    "hist_len" : 10,  # Length of the history for which past states are stored.

    # Feedback parameters
    "fb_fin": 0.1,  # Feedback parameter for financial changes.
    "fb_nonfin": 0.3,  # Feedback parameter for non-financial changes.

    # Social capital parameters
    "soc_cap_base": 0.03,  # Base level of social capital.
    "soc_cap_inf": 1,  # Influence of social capital on network dynamics.

    # Dummy variable
    "dummy": 5,  # Dummy variable.
}
