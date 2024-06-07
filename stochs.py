"""
Param bounds per parameter
"""

param_dict = {
            #'m' : [0.5, 2],
            # 'p' : [0.05, 0.2],
            # 'segregation': [0, 40],
            # 'beta' : [0.5, 20.5],
            # Event size
            "fin_event_prob": [0, 1],
            "nonfin_event_prob": [0, 1],
            # "event_size" : [0, 5],
            # Parameter indicating change of model during simulations
            "net_upd_freq" :[0, 1],
            # Pulse intervention size
            # "rec_intervention_factor" : [0, 5],
            "intervention_gap" : [5, 35],
            # History length
            "hist_len" : [1, 21],
            # Feedback parameters
            # "fb_fin": [0.05, 0.5],
            # "fb_nonfin": [0.05, 0.5],
            # soc cap parameters
            "soc_cap_base": [0.05, 0.55],
            # "soc_cap_inf": [0.5, 2],
            # Dummy variable
            # "dummy": [0, 10]
            }