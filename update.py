import numpy as np
from dynsimf.models.components.conditions.Condition import ConditionType
from dynsimf.models.components.conditions.StochasticCondition import StochasticCondition
from functions import calc_RFC, SDA_prob, calc_sens
import scipy.ndimage as ndimage
from sklearn.preprocessing import normalize
from SDA import SDA

# Update conditions
update_conditions = {
    "Network" : StochasticCondition(ConditionType.STATE, 0.01),
}
def update_states(model):
    """
    Function that updates the states each iteration
    Changes expectatations and SWB and saves current RFC
    """
    # Load states
    hab = model.get_state("habituation")
    SWB, SWB_norm, SWB_exp = model.get_state("SWB"), model.get_state("SWB_norm"), model.get_state("SWB_exp")
    fin, nonfin = model.get_state("financial"), model.get_state("nonfin")
    fin_hist, nonfin_hist, RFC_hist, SWB_hist = model.fin_hist, model.nonfin_hist, model.RFC_hist, model.SWB_hist
    N = model.constants['N']
    fin_sens, nonfin_sens = model.get_state("fin_sens"), model.get_state("nonfin_sens")
    new_fin_sens, new_nonfin_sens = fin_sens, nonfin_sens
    cur_it = len(model.simulation_output["states"])
    RFC = model.get_state("RFC")

    # Dict to save param changes to
    param_chgs = {}

    # Feedback effect of SWB delta of last time step
    SWB_delta = SWB - SWB_exp
    fin += model.constants["fb_fin"] * SWB_delta
    nonfin += model.constants["fb_nonfin"] * SWB_delta

    # Calculate expected values based on previous iteration and smoothing function
    fin_exp, nonfin_exp, RFC_exp, SWB_exp = np.empty((N)), np.empty((N)), np.empty((N)), np.empty((N))
    for i in range(N):
        fin_exp[i] = ndimage.gaussian_filter(fin_hist[i], hab[i])[-1]
        nonfin_exp[i] = ndimage.gaussian_filter(nonfin_hist[i], hab[i], )[-1]
        RFC_exp[i] = ndimage.gaussian_filter(RFC_hist[i], hab[i], )[-1]
        SWB_exp[i] = ndimage.gaussian_filter(SWB_hist[i], hab[i], )[-1]

    # Save expectations
    param_chgs["fin_exp"] = fin_exp
    param_chgs["nonfin_exp"] = nonfin_exp
    param_chgs["RFC_exp"] = RFC_exp
    param_chgs["SWB_exp"] = SWB_exp

    # Events params
    event_size = model.constants["event_size"]
    fin_event_prob = model.constants["fin_event_prob"]
    nonfin_event_prob = model.constants["nonfin_event_prob"]
    sens, desens = model.get_state('sensitisation'), model.get_state('desensitisation')

    # Calculate chance of events occuring per node
    event_chances = np.random.uniform(0, 1, (2, N))

    # Events happen
    for event_type, event_probs in enumerate(event_chances):
        # Financial event
        if event_type == 0:
            for node, p in enumerate(event_probs):
                if p < fin_event_prob:
                    event = np.random.normal(0, event_size)
                    fin[node] += event
                    new_fin_sens[node] = calc_sens([new_fin_sens[node]], [sens[node]], [desens[node]], [event], type="fin")
        # Non financial event
        else:
            for node, p in enumerate(event_probs):
                if p < nonfin_event_prob:
                    event = np.random.normal(0, event_size)
                    nonfin[node] += event
                    new_nonfin_sens[node] = calc_sens([new_nonfin_sens[node]], [sens[node]], [desens[node]], [event], type="nonfin")

    # Periodic financial interventions occurs
    rec_int_size = model.constants["rec_intervention_size"]
    if cur_it % model.constants["intervention_gap"] == 0:
        fin += rec_int_size
        new_fin_sens = calc_sens(new_fin_sens, sens, desens, np.repeat(rec_int_size, N), type="fin")

    # Set interventions occur
    int_ts = model.constants["int_ts"]
    int_size = model.constants["int_size"]
    int_type = model.constants["int_var"]
    for int_index, ts in enumerate(int_ts):
        if cur_it == ts:
            int_event = int_size[int_index]
            if int_type[int_index] == "fin":
                fin += int_event
                new_fin_sens = calc_sens(new_fin_sens, sens, desens, np.repeat(int_event, N), type="fin")
            elif int_type[int_index] == "nonfin":
                nonfin += int_event
                new_nonfin_sens = calc_sens(new_nonfin_sens, sens, desens, np.repeat(int_event, N), type="nonfin")

    fin = np.maximum(fin, 1)
    nonfin = np.maximum(nonfin, 1)

    # Calculate current RFC
    RFC_cur = calc_RFC(model)
    param_chgs["RFC"] = RFC_cur

    # Calculate relative values between previous state value and current expectation
    RFC_rel = RFC_cur / RFC_exp
    fin_rel = fin / fin_exp
    nonfin_rel = nonfin / nonfin_exp

    # Calculate sensitivity factor
    fin_sens_factor = (1/0.693147180560) * np.log(fin_sens+1)
    nonfin_sens_factor = (1/0.693147180560) * np.log(nonfin_sens+1)

    # Change SWB based on Range-Frequency comparison and financial stock
    # SWB change based on system dynamics paper
    # Divided by 2 since 2 factors now instead of 1
    RFC_SWB_change = 1 * ((1/0.693147180560) * np.log(RFC_rel+1)-1)
    fin_SWB_change = 1 * ((1/0.693147180560) * np.log(fin_rel+1)-1)
    total_fin_change = RFC_SWB_change + fin_SWB_change * fin_sens_factor
    
    # SWB change based on system dynamics paper
    nonfin_change = ((1/0.693147180560)*np.log(nonfin_rel+1)-1) * nonfin_sens_factor

    # Total change is bounded
    SWB = np.clip(SWB_norm + total_fin_change + nonfin_change, 0, 10)

    # Save new SWB, fin and nonfin values
    param_chgs["SWB"] = SWB
    param_chgs["financial"] = fin
    param_chgs["nonfin"] = nonfin

    # Save sensitivity values
    param_chgs["fin_sens"] = new_fin_sens
    param_chgs["nonfin_sens"] = new_nonfin_sens

    # Calculate and save community SWB
    # TODO fix averge community SWB
    # param_chgs["SWB_comm"] = np.array([np.mean(SWB[np.append(model.get_neighbors(node), model.get_neighbors_neighbors(node)).astype(int)]) for node in model.nodes])
    param_chgs["SWB_comm"] = 1

    # Change history
    model.fin_hist = np.append(fin_hist[:, 1:], fin.reshape(N, 1), axis=1)
    model.nonfin_hist = np.append(nonfin_hist[:, 1:], nonfin.reshape(N, 1), axis=1)    
    model.RFC_hist = np.append(RFC_hist[:, 1:], RFC_cur.reshape(N, 1), axis=1)
    model.SWB_hist = np.append(SWB_hist[:, 1:], SWB.reshape(N, 1), axis=1)

    return param_chgs

def update_network(nodes, model):
    """
    Function which changes the network by removing and adding connections
    """
    alpha = model.constants["segregation"]
    # beta = model.beta
    N = model.constants["N"]
    fin = model.get_state("financial").reshape(N, 1)

    dist = np.abs(fin - fin.T)
    probs = SDA_prob(dist, alpha, model.constants["beta"])
    matrix_size = np.shape(probs)
    # Get probability to be removed for neighbors, normalized to 1
    remove_probs = normalize(model.get_adjacency() * (1-probs), axis=1, norm='l1')
    remove_matrix = np.random.binomial(size=matrix_size, n=1, p=remove_probs)

    # Get probability to be added for non-neighbors, normalized to 1
    add_probs = normalize((1-model.get_adjacency()) * probs, axis=1, norm='l1')
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
