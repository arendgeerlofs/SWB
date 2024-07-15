import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from scipy.stats import rankdata, mannwhitneyu
import pandas as pd

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
            R_i = (I - I_min) / (I_max - I_min) # Range principle
            F_i = rankdata(fin[social_circle])[-1] / len(social_circle) # Frequency principle
            RFC_cur[node] = soc_w[node] * R_i + (1 - soc_w[node]) * F_i
    return 10 * RFC_cur

def distance(fin, nonfin, SWB):
    """
    Calculate the Euclidean distance matrix between nodes based on financial,
    non-financial, and SWB states.

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
    - event_change: Array of event change factors.
    - mode: The type of event ("fin" for financial, "nonfin" for non-financial).

    Returns:
    - An array of updated sensitivity values.
    """
    new_sens = np.empty(len(sens))
    for node, value in enumerate(sens):
        if mode == "fin":
            if event_change[node] < 1:
                rel_event_change = 1 / event_change[node]
                new_sens[node] = value / ((sens_factor[node] * rel_event_change) / 2)
            else:
                rel_event_change = event_change[node]
                new_sens[node] = value * ((desens_factor[node] * rel_event_change) / 2)
        elif mode == "nonfin":
            if event_change[node] > 1:
                rel_event_change = event_change[node]
                new_sens[node] = value * ((sens_factor[node] * rel_event_change) / 2)
            else:
                rel_event_change = 1 / event_change[node]
                new_sens[node] = value / ((desens_factor[node] * rel_event_change) / 2)
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
    init_SWB = np.clip(np.random.normal(constants["SWB_mu"], constants["SWB_sd"], constants['N']), 0.001, 100)
    return (init_fin, init_nonfin, init_SWB)

def mean_chg(data, change_point, alpha=0.05, per_agent=False):
    # Calculate if the mean has changed of a state after an intervention
    # Agent wise -> amount of agents for which it has changed
    # Mean wise -> if the average state over population has changed
    """
    Calculate if the mean has changed for a state after an intervention.

    Parameters:
    - data: The data array to analyze.
    - change_point: The point in time when the intervention occurred.
    - alpha: The significance level for the Mann-Whitney U test.
    - per_agent: Boolean indicating if the calculation is agent-wise.

    Returns:
    - chg_data: Percentage of agents which changed positive, negative and haven't changed of agent-wise
                Category indicating which way the population mean has changed or hasen't if not agent-wise
    """
    amount_agents = np.shape(data)[1]
    if per_agent:
        chg_data = np.zeros((3))
        for agent in range(amount_agents):
            segment_before, segment_after = data[:change_point, agent], data[change_point:, agent]
            _, p_value = mannwhitneyu(segment_before, segment_after)
            if p_value <= alpha:
                if np.mean(segment_before) < np.mean(segment_after):
                    chg_data[0] += 1
                else:
                    chg_data[2] += 1
            else:
                chg_data[1] += 1
        return chg_data / amount_agents
    else:
        data = np.mean(data, axis=1)
        segment_before, segment_after = data[:change_point], data[change_point:]
        _, p_value = mannwhitneyu(segment_before, segment_after)
        if p_value <= alpha:
            if np.mean(segment_before) < np.mean(segment_after):
                return 1
            else:
                return -1
        else:
            return 0

def system_behaviour_cat(chg_data):
    """
    Categorize the behavior of the system into 3 categories

    Parameters:
    - chg_data: Change data indicating the direction of mean changes per agent.

    Returns:
    - An integer representing the category of system behavior.
    """
    if chg_data[0] == 1:
        # All agents experienced positive change in mean SWB
        return 1
    elif chg_data[2] == 1:
        # All agents experienced negative change in mean SWB
        return 2
    else:
        # Other
        return 0

def get_all_data(output, params):
    """
    Extract and summarize all relevant data from the model output.

    Parameters:
    - output: The model output containing state data.
    - params: A dictionary of parameters used in the model.

    Returns:
    - results: An array of summarized data.
    """
    results = np.empty(47)

    # Save the means and variances of all states over the last 50 timesteps
    for i in range(18):
        data = extract_data(output, i)
        results[i*2] = np.mean(data[:-50])
        results[i*2 + 1] = np.var(data[:-50])

    # SWB specific data
    SWB_data = extract_data(output, 1)
    change_point = params["burn_in_period"]
    segment_before, segment_after = SWB_data[:change_point], SWB_data[change_point:]
    mean_before, mean_after = np.mean(segment_before), np.mean(segment_after)
    var_before, var_after = np.var(segment_before), np.var(segment_after)
    chg_data = mean_chg(SWB_data, change_point, per_agent=True)
    system_chg = mean_chg(SWB_data, change_point)
    system_cat = system_behaviour_cat(chg_data)

    # Save SWB specific data
    results[38] = mean_before
    results[39] = var_before
    results[40] = mean_after
    results[41] = var_after
    results[42] = system_cat
    results[43] = chg_data[0]
    results[44] = chg_data[2]
    results[45] = chg_data[1]
    results[46] = system_chg
    
    return results
