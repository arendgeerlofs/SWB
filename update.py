import numpy as np
from dynsimf.models.components.conditions.Condition import ConditionType
from dynsimf.models.components.conditions.StochasticCondition import StochasticCondition
from parameters import constants
from functions import calc_RFC, SDA_prob, calc_sens
import scipy.ndimage as ndimage
from scipy.spatial.distance import cdist
from sklearn.preprocessing import normalize
from initialise import initial_RFC_hist
from SDA import SDA

# Update conditions
update_conditions = {
    "SWB" : StochasticCondition(ConditionType.STATE, 1),
    "Event" : StochasticCondition(ConditionType.STATE, constants["event_prob"]),
    "Network" : StochasticCondition(ConditionType.STATE, 0.01),
}

def initial_RFC_update(model):
    RFC = calc_RFC(model)
    model.RFC_hist = initial_RFC_hist(model, RFC)
    return {"RFC": RFC}

# Initial updates to the model
def initial_network_update(model):
    """
    Add initial network connections using Social Distance Attachment
    - inputs -
    model : DyNSimF model object containing the graph, states and constants

    - outputs -
    dictionary containing the edge changes per node.
    Each node has an add list for which a connection is made in the network
    """
    # Get N, expected connections, alpha and financial status
    N = model.constants["N"]
    exp_con = N * model.constants["p"]
    fin = model.get_state("financial").reshape(N, 1)
    alpha = model.constants["segregation"]

    # TODO add nonfin also to distance matrix?

    # Calculate euclidean distance between nodes based on financial status
    dist = cdist(fin, fin, metric='euclidean')

    # TODO decide if we want to calculate beta?
    # sda = SDA.from_dist_matrix(D = dist, k=exp_con, alpha=alpha)
    # adj_mat = sda.adjacency_matrix(sparse=False)

    # Calculate connection probabilities
    probs = SDA_prob(dist, alpha, model.constants["beta"])

    # Create adjacency matrix and convert it to dictionary containing node pairs for edges
    adj_mat = np.random.binomial(size=np.shape(probs), n=1, p=probs)
    network_update = {}
    for node, row in enumerate(adj_mat):
        network_update[node] = {"add": []}
        for neighbor, value in enumerate(row):
            if value == 1 and node != neighbor:
                network_update[node]["add"].append(neighbor)
    
    return {'edge_change': network_update}

def update_states(model):
    """
    Function that updates the states each iteration
    Changes expectatations and SWB and saves current RFC
    """
    # Load states
    hab = model.get_state("habituation")
    SWB, SWB_norm = model.get_state("SWB"), model.get_state("SWB_norm")
    fin, nonfin = model.get_state("financial"), model.get_state("nonfin")
    fin_hist, nonfin_hist, RFC_hist, SWB_hist = model.fin_hist, model.nonfin_hist, model.RFC_hist, model.SWB_hist
    N = model.constants['N']
    fin_sens, nonfin_sens = model.get_state("fin_sens"), model.get_state("nonfin_sens")

    # Dict to save param changes to
    param_chgs = {}

    # Calculate current RFC
    RFC_cur = calc_RFC(model)
    cur_it = len(model.simulation_output["states"])
    # Save current RFC
    param_chgs["RFC"] = RFC_cur

    # Calculate expectations based on history and smoothing
    fin_exp, nonfin_exp, RFC_exp, SWB_exp = np.empty((N)), np.empty((N)), np.empty((N)), np.empty((N))
    for i in range(N):
        fin_exp[i] = ndimage.gaussian_filter(fin_hist[i], hab[i])[-1]
        nonfin_exp[i] = ndimage.gaussian_filter(nonfin_hist[i], hab[i], )[-1]
        RFC_exp[i] = ndimage.gaussian_filter(RFC_hist[i], hab[i], )[-1]
        SWB_exp[i] = ndimage.gaussian_filter(SWB_hist[i], hab[i], )[-1]

    # Calculate difference between expected and actual values
    RFC_rel = RFC_cur / RFC_exp
    fin_rel = fin / fin_exp
    nonfin_rel = nonfin / nonfin_exp

    # Change SWB based on Range-Frequency and financial stock
    # SWB change based on system dynamics paper
    # Divided by 2 since 2 factors now instead of 1
    RFC_SWB_change = 0.5 * ((1/0.693147180560) * np.log(RFC_rel+1)-1)
    fin_SWB_change = 0.5 * ((1/0.693147180560) * np.log(fin_rel+1)-1)
    total_fin_change = RFC_SWB_change + fin_SWB_change * fin_sens

    # SWB change based on system dynamics paper
    nonfin_change = ((1/0.693147180560)*np.log(nonfin_rel+1)-1) * nonfin_sens

    # Total change is bounded
    new_SWB = np.clip(SWB_norm + total_fin_change + nonfin_change, 0, 10)

    param_chgs["SWB"] = new_SWB

    # Save expectations
    param_chgs["fin_exp"] = fin_exp
    param_chgs["nonfin_exp"] = nonfin_exp
    param_chgs["RFC_exp"] = RFC_exp
    param_chgs["SWB_exp"] = SWB_exp

    # Calculate feedback effects based on SWB change with expectations
    SWB_delta = new_SWB - SWB_exp
    param_chgs["financial"] = np.maximum(fin + model.constants["fb_fin"] * SWB_delta, 1)
    param_chgs["nonfin"] = np.maximum(nonfin + model.constants["fb_nonfin"] * SWB_delta, 1)

    # Calculate and save community SWB
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

def event(nodes, model):
    """
    Functions which simulates the occurance of an event to the financial status of nodes
    """
    # Get financial status and event size
    fin = model.get_state("financial")
    event_size = model.constants["event_size"]
    fin_sens = model.get_state('fin_sens')
    sens, desens = model.get_state('sensitisation'), model.get_state('desensitisation')

    # Calculate event
    change = np.random.normal(0, event_size, len(nodes))

    new_fin_sens = calc_sens(fin_sens[nodes], sens[nodes], desens[nodes], change / event_size, type=1)
    return {"financial": np.maximum(fin[nodes] + change, 1), 'fin_sens': np.clip(new_fin_sens, 0.25, 4)}

def pulse(model_input, model):
    """
    Returns all nodes if intervention takes place
    """
    cur_it = len(model.simulation_output["states"])
    if cur_it % model.constants["intervention_gap"] == 0:
        return model_input[0]
    return np.array([])

def set_pulse(model_input, model, it):
    cur_it = len(model.simulation_output["states"])
    if cur_it == it:
        return model_input[0]  
    return np.array([])  

def intervention(model):
    """
    Perform intervention on all nodes
    """
    fin = model.get_state("financial")
    int_size = model.constants["intervention_size"]
    event_size = model.constants["event_size"]
    fin_sens = model.get_state('fin_sens')
    sens, desens = model.get_state('sensitisation'), model.get_state('desensitisation')

    new_fin_sens = calc_sens(fin_sens, sens, desens, [int_size / event_size], type=1)
    return {"financial": fin + int_size, 'fin_sens': np.clip(new_fin_sens, 0.25, 4)}
