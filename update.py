import numpy as np
import scipy.ndimage as ndimage
from sklearn.preprocessing import normalize
from functions import calc_RFC, SDA_prob, calc_sens, distance, calc_soc_cap

def update_states(model):
    """
    Updates the states for each iteration.
    This function modifies the expectations and subjective well-being (SWB) of nodes,
    and saves the current Relative Financial Capital (RFC).

    Parameters:
    - model: The model object containing the current state and constants.

    Returns:
    - param_chgs: Dictionary containing updated state parameters.
    """
    # Load states
    hab = model.get_state("habituation")
    SWB = model.get_state("SWB")
    SWB_norm = model.get_state("SWB_norm")
    SWB_exp = model.get_state("SWB_exp")
    fin = model.get_state("financial")
    nonfin = model.get_state("nonfin")
    fin_hist = model.fin_hist
    nonfin_hist = model.nonfin_hist
    SWB_hist = model.SWB_hist
    RFC_hist = model.RFC_hist
    soc_cap_hist = model.soc_cap_hist
    N = model.constants['N']
    fin_sens = model.get_state("fin_sens")
    nonfin_sens = model.get_state("nonfin_sens")
    new_fin_sens = fin_sens.copy()
    new_nonfin_sens = nonfin_sens.copy()
    cur_it = len(model.simulation_output["states"])
    soc_cap = model.get_state("soc_cap")

    # Dict to save parameter changes to
    param_chgs = {}

    # Feedback effect of SWB delta of last time step
    SWB_rel = SWB / SWB_exp
    for node, node_rel in enumerate(SWB_rel):
        if SWB_rel[node] < 1:
            fin[node] *= 1 - (1 - node_rel) * model.constants["fb_fin"]
            nonfin[node] *= 1 - (1- node_rel) * model.constants["fb_nonfin"]
        else:
            fin[node] *= 1 + (1 -( 1 / node_rel)) * model.constants["fb_fin"]
            nonfin[node] *= 1 + (1 - (1 / node_rel)) * model.constants["fb_nonfin"]

    # Calculate expected values based on previous iteration and smoothing function
    fin_exp, nonfin_exp, RFC_exp, soc_cap_exp, SWB_exp = np.empty(N), np.empty(N), np.empty(N), np.empty(N), np.empty(N)
    for i in range(N):
        fin_exp[i] = ndimage.gaussian_filter1d(fin_hist[i], hab[i])[-1]
        nonfin_exp[i] = ndimage.gaussian_filter1d(nonfin_hist[i], hab[i])[-1]
        RFC_exp[i] = ndimage.gaussian_filter1d(RFC_hist[i], hab[i])[-1]
        soc_cap_exp[i] = ndimage.gaussian_filter1d(soc_cap_hist[i], hab[i])[-1]
        SWB_exp[i] = ndimage.gaussian_filter1d(SWB_hist[i], hab[i])[-1]

    # Save expectations
    param_chgs["fin_exp"] = fin_exp
    param_chgs["nonfin_exp"] = nonfin_exp
    param_chgs["RFC_exp"] = RFC_exp
    param_chgs["soc_cap_exp"] = soc_cap_exp
    param_chgs["SWB_exp"] = SWB_exp

    # Events parameters
    event_size = model.constants["event_size"]
    fin_event_prob = model.constants["fin_event_prob"]
    nonfin_event_prob = model.constants["nonfin_event_prob"]
    soc_cap_base = model.constants["soc_cap_base"]
    soc_cap_inf = model.constants["soc_cap_inf"]
    sens = model.get_state('sensitisation')
    desens = model.get_state('desensitisation')

    # Calculate chance of events occurring per node
    event_chances = np.random.uniform(0, 1, (2, N))

    print(f"-----{cur_it}-----")
    # Burn in period
    if cur_it > model.constants["burn_in_period"]:
        # Events happen
        for event_type, event_probs in enumerate(event_chances):
            if event_type == 0:  # Financial event
                for node, p in enumerate(event_probs):
                    if p < fin_event_prob:
                        event = np.random.normal(0, event_size)
                        rel_chg = event / fin[node]
                        if rel_chg < 0:
                            rel_chg = 1 / -(rel_chg - 1)
                        else:
                            rel_chg += 1
                        fin[node] *= rel_chg
                        new_fin_sens[node] = calc_sens([new_fin_sens[node]], [sens[node]], [desens[node]], np.array([rel_chg]), mode="fin")
            else:  # Non-financial event
                for node, p in enumerate(event_probs):
                    if p < nonfin_event_prob:
                        event = np.random.normal(0, event_size)
                        rel_chg = event / nonfin[node]
                        if rel_chg < 0:
                            rel_chg = 1 / -(rel_chg - 1)
                        else:
                            rel_chg += 1
                        nonfin[node] *= rel_chg
                        new_nonfin_sens[node] = calc_sens([new_nonfin_sens[node]], [sens[node]], [desens[node]], np.array([rel_chg]), mode="nonfin")

        # Periodic financial interventions occur
        rec_int_factor = model.constants["rec_intervention_factor"]
        if (cur_it - model.constants["burn_in_period"]) % model.constants["intervention_gap"] == 0:
            print("Intervention")
            fin *= rec_int_factor
            new_fin_sens = calc_sens(new_fin_sens, sens, desens, np.repeat(rec_int_factor, N), mode="fin")

        # Set interventions occur
        int_ts = model.constants["int_ts"]
        int_size = model.constants["int_size"]
        int_type = model.constants["int_var"]
        for int_index, ts in enumerate(int_ts):
            if cur_it == ts:
                int_event = int_size[int_index]
                if int_type[int_index] == "fin":
                    fin *= int_event
                    new_fin_sens = calc_sens(new_fin_sens, sens, desens, np.repeat(int_event, N), mode="fin")
                elif int_type[int_index] == "nonfin":
                    nonfin *= int_event
                    new_nonfin_sens = calc_sens(new_nonfin_sens, sens, desens, np.repeat(int_event, N), mode="nonfin")

    fin = np.maximum(fin, 0.001)
    nonfin = np.maximum(nonfin, 0.001)

    # Calculate current RFC
    RFC = calc_RFC(model)
    param_chgs["RFC"] = RFC

    # Calculate current social capital
    soc_cap = calc_soc_cap(model)
    param_chgs["soc_cap"] = soc_cap

    RFC = np.maximum(RFC, 0.001)
    soc_cap = np.maximum(soc_cap, 0.001)

    fin_exp = np.maximum(fin_exp, 0.001)
    nonfin_exp = np.maximum(nonfin_exp, 0.001)
    RFC_exp = np.maximum(RFC_exp, 0.001)
    soc_cap_exp = np.maximum(soc_cap_exp, 0.001)
    
    # Calculate relative values between previous state value and current expectation
    fin_rel = fin / fin_exp
    # print(np.mean(fin_rel))
    nonfin_rel = nonfin / nonfin_exp
    RFC_rel = RFC / RFC_exp
    soc_cap_rel = soc_cap / soc_cap_exp
    
    # Calculate sensitivity factor
    # fin_sens_factor = (1 / np.log(2)) * np.log(fin_sens + 1)
    # nonfin_sens_factor = (1 / np.log(2)) * np.log(nonfin_sens + 1)

    # Change SWB based on Range-Frequency comparison and financial stock
    RFC_SWB_change = (1 / np.log(2)) * np.log(RFC_rel + 1) - 1
    fin_SWB_change = ((1 / np.log(2)) * np.log(fin_rel + 1) - 1) * fin_sens
    # print(np.mean(fin_SWB_change))
    total_fin_change = RFC_SWB_change + fin_SWB_change

    # SWB change based on system dynamics paper
    soc_cap_change = (1 / np.log(2)) * np.log(soc_cap_rel + 1) - 1
    nonfin_change = ((1 / np.log(2)) * np.log(nonfin_rel + 1) - 1) * nonfin_sens
    total_nonfin_change = soc_cap_change + nonfin_change
    # print(np.mean(total_fin_change), np.mean(total_nonfin_change))

    # Social resillience
    SWB_change = total_fin_change + total_nonfin_change
    for i, node_change in enumerate(SWB_change):
        if node_change < 0:
            SWB_change[i] = node_change / ((soc_cap[i] / soc_cap_base) * soc_cap_inf)
    # print(np.mean(SWB_change))

    # Bound SWB
    SWB = np.clip(SWB_norm + SWB_change, 0.001, 100)

    # Save new SWB, fin and nonfin values
    param_chgs["SWB"] = SWB
    param_chgs["financial"] = fin
    param_chgs["nonfin"] = nonfin

    # Save sensitivity values
    param_chgs["fin_sens"] = new_fin_sens
    param_chgs["nonfin_sens"] = new_nonfin_sens

    # Calculate and save community SWB
    # TODO: Fix average community SWB
    param_chgs["SWB_comm"] = 1

    # Save degrees
    param_chgs["degrees"] = np.sum(model.get_adjacency(), axis=1)

    # Change history
    model.fin_hist = np.append(fin_hist[:, 1:], fin.reshape(N, 1), axis=1)
    model.nonfin_hist = np.append(nonfin_hist[:, 1:], nonfin.reshape(N, 1), axis=1)    
    model.RFC_hist = np.append(RFC_hist[:, 1:], RFC.reshape(N, 1), axis=1)
    model.soc_cap_hist = np.append(soc_cap_hist[:, 1:], soc_cap.reshape(N, 1), axis=1)
    model.SWB_hist = np.append(SWB_hist[:, 1:], SWB.reshape(N, 1), axis=1)

    return param_chgs

def update_network(nodes, model, network_type="SDA"):
    """
    Updates the network by removing and adding connections based on calculated probabilities.
    
    Parameters:
    - nodes: List of nodes in the network.
    - model: The model object containing the current state and constants.

    Returns:
    - A dictionary with the changes to the network, specifying edges to be removed and added.
    """
    # Number of nodes
    N = model.constants["N"]

    # Adjacency matrix
    adj_mat = model.get_adjacency()

    # Size of adjacency matrix
    matrix_size = np.shape(adj_mat)

    # Probability placeholders
    add_probs = np.empty(0)
    remove_probs = np.empty(0)
    
    if network_type == "SDA":
        # Segregation parameter from model constants
        alpha = model.constants["segregation"]
        
        # Get the current financial, non-financial, and SWB states
        fin = model.get_state("financial").reshape(N, 1)
        nonfin = model.get_state("nonfin").reshape(N, 1)
        SWB = model.get_state("SWB").reshape(N, 1)
        
        # Calculate the distance matrix based on financial, non-financial, and SWB states
        dist = distance(fin, nonfin, SWB)
        
        # Calculate the probability matrix for changes in the network
        probs = SDA_prob(dist, alpha, model.constants["beta"])
        
        # Get probability to be removed for neighbors, normalized to 1
        remove_probs = normalize(adj_mat * (1 - probs), axis=1, norm='l1')
        
        # Get probability to be added for non-neighbors, normalized to 1
        add_probs = normalize((1 - adj_mat) * probs, axis=1, norm='l1')
        
    elif network_type == "Rd":
        # Get probability to be removed for neighbors, with total per node equal to 1
        remove_probs = adj_mat / np.sum(adj_mat, axis=1).reshape(N, 1)

        # Get probability to be added for neighbors, with total equal to 1
        add_probs = (1-adj_mat) / np.sum((1-adj_mat), axis=1).reshape(N, 1)

    else:
        # Get probability to be removed for neighbors, with total per node equal to 1
        # Probability formula of Barabasi-Albert from https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model
        rem_degree_mat = (np.sum((1-adj_mat), axis=1) * (adj_mat))
        remove_probs = rem_degree_mat / np.sum(rem_degree_mat, axis=1).reshape(N, 1)

        # Get probability to be added for neighbors, with total equal to 1
        add_degree_mat = (np.sum(adj_mat, axis=1) * (1-adj_mat))
        add_probs = add_degree_mat / np.sum(add_degree_mat, axis=1).reshape(N, 1)

    # Compute connection to be added and removed
    remove_matrix = np.random.binomial(size=matrix_size, n=1, p=remove_probs)
    add_matrix = np.random.binomial(size=matrix_size, n=1, p=add_probs)

    network_update = {}
    for node in nodes:
        network_update[node] = {"remove": [], "add": []}
        for neighbor in range(matrix_size[0]):
            if remove_matrix[node, neighbor] == 1:
                network_update[node]["remove"].append(neighbor)
            if add_matrix[node, neighbor] == 1:
                network_update[node]["add"].append(neighbor)  
    return {'edge_change': network_update}
