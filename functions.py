import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from scipy.stats import rankdata, mannwhitneyu

def calc_RFC(model):
    """
    Calculate the Range-Frequency Compromise (RFC) for each node in the network.

    Parameters:
    - model: The model object containing the current state and constants.

    Returns:
    - RFC_cur: An array of RFC values for each node.

    Reference
    - Allen Parducci. Happiness, pleasure, and judgment: The contextual theory and its
    applications. Lawrence Erlbaum Associates, Inc, 1995
    """
    N = model.constants["N"]
    fin = model.get_state("financial")
    soc_w = model.get_state("soc_w") / 10
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
            # Range principle
            if (I_max - I_min) == 0:
                print(f"{I_max}----{I_min}")
            R_i = (I - I_min) / (I_max - I_min)
            # Frequency principle
            F_i = rankdata(fin[social_circle])[-1] / len(social_circle)
            RFC_cur[node] = soc_w[node] * R_i + (1 - soc_w[node]) * F_i
    return 10 * RFC_cur

def distance(fin, nonfin, SWB):
    """
    Calculate the Euclidean distance matrix between nodes based on financial, non-financial, and SWB states.

    Parameters:
    - fin: Array of financial states.
    - nonfin: Array of non-financial states.
    - SWB: Array of SWB (Subjective Well-Being) states.

    Returns:
    - dist: A matrix of Euclidean distances between nodes.
    """
    character = np.column_stack((fin, nonfin, SWB))
    dist = cdist(character, character, metric='euclidean')
    return dist

def calc_soc_cap(model):
    """
    Calculate the social capital for each node in the network.

    Parameters:
    - model: The model object containing the current state and constants.

    Returns:
    - An array of social capital values for each node.
    """
    N = model.constants["N"]
    fin = model.get_state("financial").reshape(N, 1)
    nonfin = model.get_state("nonfin").reshape(N, 1)
    SWB = model.get_state("SWB").reshape(N, 1)
    dist = distance(fin, nonfin, SWB)
    likeness = 1 - normalize(dist, norm="max", axis=1)
    adj_mat = model.get_adjacency()
    return np.maximum(np.mean(likeness * adj_mat, axis=1), 0.0001)

def SDA_prob(dist, alpha, beta):
    """
    Calculate the probability of connection changes based on distance and model parameters.

    Parameters:
    - dist: A matrix of distances between nodes.
    - alpha: Segregation parameter.
    - beta: Model parameter.

    Returns:
    - A matrix of probabilities for connection changes.
    """
    return 1 / (1 + (beta ** (-1) * dist) ** alpha)

def extract_data(output, state_id):
    """
    Extract the data from the model output for all nodes and return as a 3D array.

    Parameters:
    - nodes: The number of nodes.
    - output: The model output containing state data.
    - state_id: The index of the state to extract.

    Returns:
    - A 3D array of extracted data.
    """
    node_amount = np.shape(output["states"][0])[0]
    data = np.zeros((len(output["states"]), node_amount))
    for timestep in output["states"]:
        data[timestep] = output["states"][timestep][:, state_id]
    return data

def calc_sens(sens, sens_factor, desens_factor, event_change, mode="fin"):
    """
    Calculate the new sensitivity values based on event changes.

    Parameters:
    - sens: Array of current sensitivity values.
    - sens_factor: Array of sensitization factors.
    - desens_factor: Array of desensitization factors.
    - event_change: Array of event changes.
    - mode: The type of event ("fin" for financial, "nonfin" for non-financial).

    Returns:
    - An array of updated sensitivity values.
    """
    new_sens = np.empty(len(sens))
    rel_event_change = event_change - 1
    for node, value in enumerate(sens):
        if mode == "fin":
            if event_change[node] > 1:
                new_sens[node] = value * (1 + (sens_factor[node] * rel_event_change[node])/2)
            else:
                new_sens[node] = value * (1 + (desens_factor[node] * rel_event_change[node])/2)
        elif mode == "nonfin":
            if event_change[node] > 0:
                new_sens[node] = value * (1 + (sens_factor[node] * rel_event_change[node])/2)
            else:
                new_sens[node] = value / (1 + (desens_factor[node] * rel_event_change[node])/2)
    return np.clip(new_sens, 0.25, 4)

def init_ind_params(constants):
    """
    Initialize individual parameters for the model.

    Parameters:
    - constants: A dictionary of constants used in the model.

    Returns:
    - init_fin: Initial financial states.
    - init_nonfin: Initial non-financial states.
    - init_SWB: Initial SWB (Subjective Well-Being) states.
    """
    init_fin = np.random.uniform(10, 100, constants['N'])
    init_nonfin = np.random.uniform(10, 100, constants['N'])
    init_SWB = np.clip(np.random.normal(constants["SWB_mu"], constants["SWB_sd"], constants['N']), 0.001, 10)
    return (init_fin, init_nonfin, init_SWB)

def mean_chg(data, change_point, alpha=0.05, per_agent=False):
    # Calculate if the mean has changed of a state after an intervention
    # Agent wise -> amount of agents for which it has changed
    # Mean wise -> if the average state over population has changed
    amount_agents = np.shape(data)[1]
    if per_agent:
        amount_changed = 0
        for agent in range(amount_agents):
            segment_before, segment_after = data[:change_point, agent], data[change_point:, agent]
            _, p_value = mannwhitneyu(segment_before, segment_after)
            if p_value <= alpha:
                amount_changed += 1
        return amount_changed / amount_agents
    else:
        segment_before, segment_after = data[:change_point], data[change_point:]
        _, p_value = mannwhitneyu(segment_before, segment_after)
        if p_value <= alpha:
            return True
        
def is_oscillatory(output):
    return

def system_behaviour_cat(data, params):
    "Categorises the behaviour of the system into 6 categories"
    chg_point = params["burn_in_period"]
    amount_chgd = mean_chg(data, chg_point, per_agent=True)
    if is_oscillatory(data[chg_point:]):
        # There is oscillatory behaviour around the SWB equilirbrium of the agents (unstable system)
        return 0
    elif amount_chgd == 0:
        # No agents equilibrium SWB changed
        return 1
    elif amount_chgd < 1/3:
        # 0 and 2/3rd's of the agents changed SWB
        return 2
    elif amount_chgd < 2/3:
        # between 1/3rd's and 2/3rd's of the agents changed SWB
        return 3
    elif amount_chgd < 1:
        # 2/3rd's or more but not all of the agents changed SWB
        return 4
    else:
        # All agents changed their SWB
        return 5
