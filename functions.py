import numpy as np
from scipy.stats import rankdata

def calc_RFC(model):
    N = model.constants["N"]
    fin = model.get_state("financial")
    soc_w  = model.get_state("soc_w")/10
    RFC_cur = np.empty(N)
    for node in model.nodes:
        I = fin[node]
        neighbors = model.get_neighbors(node)
        social_circle = np.append(neighbors, node).astype(int)
        I_min = np.min(fin[social_circle])
        I_max = np.max(fin[social_circle])
        if I_min == I_max:
            RFC_cur[node] = 0.5
        else:
            R_i = (I - I_min)/(I_max-I_min)
            F_i = rankdata(fin[social_circle])[-1]/len(social_circle)
            RFC_cur[node] = soc_w[node] * R_i + (1-soc_w[node])*F_i
    return 10* RFC_cur

def SDA_prob(dist, alpha, beta):
    return 1 / (1+(beta**(-1)*dist)**alpha)

def extract_data(nodes, output, state_number):
    """
    Extract the data from the model output for all nodes and return as a 3d array
    """
    data = np.zeros((len(output["states"]), nodes))
    for timestep in output["states"]:
        data[timestep] = output["states"][timestep][:, state_number]
    return data

def calc_sens(sens, sens_factor, desens_factor, event_change, type="fin"):
    new_sens = np.empty(len(sens))
    for node, value in enumerate(sens):
        if type == "fin":
            if event_change[node] > 0:
                new_sens[node] = value / (1 + ((sens_factor[node] * event_change[node]) / 10))
            else:
                new_sens[node] = value * (1 + (-(desens_factor[node] * event_change[node]) / 10))
        elif type == "nonfin":
            if event_change[node] > 0:
                new_sens[node] = value * (1 + ((sens_factor[node] * event_change[node]) / 10))
            else:
                new_sens[node] = value / (1 + (-(desens_factor[node] * event_change[node]) / 10))
    return new_sens


