import copy
import itertools
import numpy as np
from tqdm import tqdm
from numba import jit
import collections


class NumbaMCMC:
    def __init__(self, qubo, n_steps=10 ** 4, interval=1):
        """
        MCMC class

        Attributes
        ----------
        qubo : np array
            QUBO as triangle matrix
        n_steps: int
            steps for mcmc
        interval: int
            interval steps for logging
        """

        self.dim = qubo.shape[0]
        self.n_steps = n_steps

        # init state
        self.state = np.ascontiguousarray(np.zeros(self.dim))
        self.current_E = 0.0

        # make symmetrical qubo to calc dE
        self.s_qubo = np.ascontiguousarray(
            prep_s_qubo(qubo, self.dim)).astype(np.float64)

        # log
        self.log_E = []
        self.log = []

        self.current_step = 0
        self.interval = interval

    def run(self):
        """
        run mcmc for n_steps
        """
        for i in tqdm(range(self.n_steps)):
            self.step()

    def step(self):
        """
        one step mcmc
        """
        # step
        self.state, self.current_E = step(
            self.state, self.current_E, self.dim, self.s_qubo)
        # log

        self.current_step += 1

        if self.current_step % self.interval == 0:
            self.log_E.append(self.current_E)
            self.log.append(self.state)


@jit
def prep_s_qubo(qubo, dim):
    """
    prepare symmetrical qubo

    Parameters
    ----------
    qubo : np array
        qubo matrix
    dim: int
        size of qubo
    """
        
    s_qubo = qubo.copy()
    for i in range(dim):
        for j in range(dim):
            if i <= j:
                continue
            s_qubo[i][j] = qubo[j][i]

    return s_qubo


@jit
def step(state, E, dim, s_qubo):
    """
    one mcmc step

    Parameters
    ----------
    state : np array
        current state (101011....)
    E: float
        current energy
    dim: int
        size of qubo
    s_qubo: np array
        symmetrical qubo
        
    Returns
    -------
    next_state : np array
        state after one step
    next_E: float
        energy after one step

    """
    
    i = np.random.randint(0, dim)

    cand_state = state.copy()

    # calc delta E
    if state[i]:
        dE = -np.dot(s_qubo[i, :], state)
        cand_state[i] = 0
    else:
        cand_state[i] = 1
        dE = np.dot(s_qubo[i, :], cand_state)

    # transfer state
    if dE < 0:
        next_state = cand_state
        next_E = E+dE
    else:
        # change state
        if np.random.rand() <= np.exp(-dE):
            next_state = cand_state
            next_E = E+dE
        else:
            next_state = state
            next_E = E

    return next_state, next_E


def counter(narr, prob_mode=False):
    dis = []
    for i in range(len(narr)):
        dis.append("".join([str(int(x)) for x in narr[i]]))

    ccc = collections.Counter(dis)
    if not prob_mode:
        return ccc
    else:
        tot = sum(list(ccc.values()))
        probDict = {}
        for k, v in ccc.items():
            probDict[k] = v / tot
        return probDict
