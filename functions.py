import numpy as np
from scipy.stats import rankdata

def calc_RFC(model):
    N = model.constants["N"]
    fin = model.get_state("financial")
    w = 0.5
    RFC_cur = np.empty(N)
    for node in model.nodes:
        I = fin[node]
        neighbors = model.get_neighbors(node)
        social_circle = np.append(neighbors, node)
        I_min = np.min(fin[social_circle])
        I_max = np.max(fin[social_circle])
        R_i = (I - I_min)/(I_max-I_min)
        F_i = rankdata(fin[social_circle])[-1]/len(social_circle)
        RFC_cur[node] = w * R_i + (1-w)*F_i
    return RFC_cur

def get_nodes(graph):
    return graph.nodes

def SDA_root(exp_con, N, dist, alpha, beta):
    probs = SDA_prob(dist, alpha, beta)
    return exp_con - (1/N) * np.sum(probs)



def SDA_prob(dist, alpha, beta):
    return 1 / (1+(beta**(-1)*dist)**alpha)

def bisection(f, exp_con, N, dist, alpha, b_min, b_max, tol): 
    # approximates a root, R, of f bounded 
    # by a and b to within tolerance 
    # | f(m) | < tol with m the midpoint 
    # between a and b Recursive implementation
    b_min_sign = np.sign(f(exp_con, N, dist, alpha, b_min))
    b_max_sign = np.sign(f(exp_con, N, dist, alpha, b_max))
    # check if a and b bound a root
    if b_min_sign == b_max_sign:
        raise Exception(
         "The scalars a and b do not bound a root")
        
    # get midpoint
    m = (b_min + b_max)/2

    b_mid_sign = np.sign(f(exp_con, N, dist, alpha, m))
    
    if np.abs(f(exp_con, N, dist, alpha, m)) < tol:
        # stopping condition, report m as root
        return m
    elif b_min_sign == b_mid_sign:
        # case where m is an improvement on a. 
        # Make recursive call with a = m
        return bisection(f, exp_con, N, dist, alpha, m, b_max, tol)
    elif b_max_sign == b_mid_sign:
        # case where m is an improvement on b. 
        # Make recursive call with b = m
        return bisection(f, exp_con, N, dist, alpha, b_min, m, tol)