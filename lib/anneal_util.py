import random
import numpy as np
from NumbaMCMC import NumbaMCMC


def submit_anneal_job(qubo,n_steps=10**7):
    """
    conduct annealing (using MCMC)  *DAU is available only on private server

    Parameters
    ----------
    qubo : np array
        qubo
    n_steps : int
        steps for mcmc

    Returns
    -------
    state_list : list
        sampled solutions
    energy_list : list
        their energy
    c_list: list
        scaling factor
    """
    
    state_list=[]
    energy_list=[]
    c_list=[]

    for c in [1,10,100]:
        mcmc=NumbaMCMC(-qubo*c,n_steps=n_steps,interval=10**4)
        mcmc.run()
        state_list.extend(mcmc.log)
        energy_list.extend(list(np.array(mcmc.log_E)/c))
        c_list.extend([c]*len(mcmc.log))

    return state_list,energy_list,c_list



def mix_qubo_sampling(c1, c2, rbm_qubo, model_qubo,
                      sample_num=16):
    """
    conduct annealing from two QUBOs

    Parameters
    ----------
    c1 : float
        weight for the first qubo during sampling
    c2 : float
        weight for the second qubo during sampling
    rbm_qubo : np array
        fist qubo (e.g, made by rbm)
    model_qubo : np array
        second qubo (e.g, made by regression)
    sample_num: int
        number of final outputted binary solutions (randomly selected)

    Returns
    -------
    res_dict : dict
        dict of sampled data
    """

    # set anneal condition
    pad_coef = np.pad(model_qubo, [0, rbm_qubo.shape[0]-model_qubo.shape[0]])
    reg_qubo = np.diag(pad_coef)
    qubo = -c1*reg_qubo-c2*rbm_qubo

    # anneal
    found_fp_list = submit_anneal_job(qubo)

    # process annealed results
    found_fp_list = found_fp_list[:, :model_qubo.shape[0]]

    selected_fp_list = random.sample(list(found_fp_list), sample_num)

    res_dict = {}
    res_dict["fp_list"] = found_fp_list
    res_dict["found_fp_list"] = selected_fp_list
    res_dict["reg_qubo*fp_list"] = np.dot(
        np.array(res_dict["fp_list"]), model_qubo)
    return res_dict


# annealer: 2021524_anneal_const_parallel
def r_qubo_sampling(r, rbm_qubo, model_qubo,
                    sample_num=16):
    """
    conduct annealing from two QUBOs

    Parameters
    ----------
    r : float
        weight ratio for the first and second QUBOs
    rbm_qubo : np array
        fist qubo (e.g, made by rbm)
    model_qubo : np array
        second qubo (e.g, made by regression)
    sample_num: int
        number of final outputted binary solutions (randomly selected)

    Returns
    -------
    res_dict : dict
        dict of sampled data
    """

    c1 = r
    c2 = 1/r
    pad_coef = np.pad(model_qubo, [0, rbm_qubo.shape[0]-model_qubo.shape[0]])
    reg_qubo = np.diag(pad_coef)
    qubo = -c1*reg_qubo-c2*rbm_qubo

    # anneal
    return submit_anneal_job(qubo)


def random_state_sampling(state_list, eg_list, n_sampling=20):
    """
    randomly select solutions according to their enegy

    Parameters
    ----------
    state_list : list of int list
        list of fingerprint
    eg_list : list of float
        list of energy for state_list
    n_sampling: int
        number of final outputted binary solutions (randomly selected)

    Returns
    -------
    sel_ids : list of int
        selected ids
    """

    sel_ids = []

    scale = 10**10
    eg_array = np.array(eg_list)*scale
    emin = min(eg_array)
    emax = max(eg_array)

    # min energy
    cid = np.where(eg_array == emin)[0][0]
    sel_ids.append(cid)

    # random select
    while len(sel_ids) < n_sampling:
        rand_eg = np.random.randint(emin, emax, 1)
        dist = abs(eg_array-rand_eg)
        cid = np.where(dist == min(dist))[0][0]
        sel_ids.append(cid)
        sel_ids = list(set(sel_ids))

    return sel_ids
