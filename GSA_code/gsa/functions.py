import sys, os, subprocess, h5py
import numpy as np

# Input definition functions
def settings(n_xi: int, n_eta: int=1, n_reps: int=None, seed: int=None, returnDeck=False):
    """ Define gsa settings and instantiate the random number generator.

    input_deck is an instance of the InputDeck class (below). 
    If the user chooses a seed, the rng for sample creation will use that seed;
    otherwise, the seeding is random. Either way, the seed is saveable as a setting
    for future reproduction.    

    Parameters
    ------------
        n_xi: int
            Number of samples for each matrix such that the shape of all 
            matrices is [n_xi, k], where k is known from the number of defined 
            parameters. The total number of simulations will be n_xi(k+2). 
        n_eta: int
            Number of samples used for each stochastic solver run.
        n_reps: int, optional (default None)
            Number of repetitions to compute Si and Ti are computed for 
            statistics.
        seed: int, optional (default None)
            Seed to instantiate rng instance. If None, seeding is random.
            128-bit integer recommended; see https://numpy.org/doc/stable/reference/random/bit_generators/index.html#seeding-and-entropy
    """
    # Rng instantiation for parameter-space sampling
    ss = np.random.SeedSequence(seed)
    ss = np.random.SeedSequence(int(np.array([ss.entropy]).astype(np.float64)))
    input_deck.seed = ss.entropy                    # If user has set seed, entropy returns that seed. Otherwise, it"s the seed of the instantiated rng
    input_deck.rng = np.random.default_rng(ss)                  # Carry instantiated random generator
    input_deck.init_seed = ss.entropy

    input_deck.n_xi = int(n_xi)
    input_deck.n_eta = int(n_eta)
    if n_reps is not None:
        input_deck.n_reps = int(n_reps)

    if returnDeck:
        return input_deck    


def parameter(key=None, dist="uniform", pmin=None, pmax=None, avg=None, dev=None, mu=None, sigma_sq=None):
    """ Define an input parameter and its range.

    User must provide either p_min and p_max or avg and dev.

    Parameters
    ------------
        key: string
            Keyword to be replaced with a sampled value in generate_inputs()
        dist: float, optional (default: "uniform")
            Parameter's distribution. Current options: uniform, normal
            Uniform requires either pmin and pmax or avg and dev.
            Normal requires both mu and sigma_sq.
        pmin: float, optional
            Parameter minimum; used for uniform distribution.
        pmax: float, optional
            Parameter maximum; used for uniform distribution.
        avg: float, optional
            Parameter average; used for uniform distribution.
        dev: float, optional
            Parameter deviation from avg; used for uniform distribution.
        mu: float, optional
            Parameter average; used for normal distribution.
        sigma_sq: float, optional
            Parameter variance; used for normal distribution.

    """
    if dist == "uniform":
        if pmin is None:
            if pmax is not None:
                print_error("Either pmin and pmax or avg and dev must be supplied for uniform distribution.")
            elif None in {avg, dev}:
                print_error("Either pmin and pmax or avg and dev must be supplied for uniform distribution.")
            else:
                pmin = avg - dev
                pmax = avg + dev
        param = {"a": pmin, "b": pmax}
    elif dist == "normal":
        if None in {mu, sigma_sq}:
            print_error("Both mu and sigma_sq must be supplied for normal distribution.")
        param = {"a": mu, "b": sigma_sq}
    param["key"] = key
    param["dist"] = dist
    input_deck.parameters.append(param)
    input_deck.k += 1


def generate_samples(outfile=None, gendir="generated/", genseeds=False, returnDeck=False): 
    """ Generate A and B matrices, plus A_B matrix for every parameter.

    Parameters
    ------------
        outfile: string, optional (default: None)
            Name of file to which generated samples are written. 
            By default, the A and B matrices are an attribute of input_deck.
            If provided, all matrices and the rng seed used to generate them are
            written to outfile. 
        gendir: string, optional (default: "generated/")
            Directory to which outfile is written.
        genseeds: bool, optional (default: False)
            If True, generate list of seeds to pass to stochastic solver
    """
    k = input_deck.k
    nxi = input_deck.n_xi
    A = np.zeros((nxi, k))
    B = np.zeros((nxi, k))

    for p_id in range(k):
        parameter = input_deck.parameters[p_id]
        if parameter["dist"] == "uniform":
            xi = input_deck.rng.uniform(low=parameter['a'], high=parameter['b'], size=2*nxi)
        elif parameter["dist"] == "normal":
            xi = input_deck.rng.normal(loc=parameter['a'], scale=parameter['b'], size=2*nxi)
        A[:, p_id] = xi[:nxi]
        B[:, p_id] = xi[nxi:]
    input_deck.matrix["A"] = A
    input_deck.matrix["B"] = B

    if outfile is not None:
        if not os.path.isdir(gendir):
            subprocess.run(["mkdir", gendir])
        outfile = gendir + outfile
        AB = np.zeros((k, input_deck.n_xi, k))
        for param_id in range(k):
            AB[param_id, :, :] = A.copy()
            AB[param_id, :, param_id] = B[:, param_id].copy()
        f = h5py.File(outfile, "w")
        f.create_dataset(name="A", data=A)
        f.create_dataset(name="B", data=B)
        f.create_dataset(name="AB", data=AB)
        f.create_dataset(name="seed", data=np.array([input_deck.init_seed]).astype(np.float64))
        if genseeds:
            seed = input_deck.rng.integers(2**63, size=(k+2)*input_deck.n_xi)
            f.create_dataset(name="solver_seeds", data=seed)
        f.close()
        
    if returnDeck:
        return input_deck    


# Calculate Sobol' indices
def compute_saltelli_indices(results):
    """ Computes first-order indices from Sobol' (2007) and total-order from Jansen (1999)
        [0] is standard, [1] is vard
    """
    fA, fB, fAB, sigsq_A, sigsq_B, sigsq_AB = read_results(results)
    datashape = (2,) + fA.shape[1:]
    vi = np.zeros((input_deck.k,) + datashape)
    eni = vi.copy()

    var = var_AB(fA, fB, sigsq_A, sigsq_B)


    for i in range(input_deck.k):
        fABi = fAB[i]
        sigsq_ABi = sigsq_AB[i]
        vi[i] = vi_saltelli(fA, fB, fABi)
        eni[i] = eni_jansen(fA, fABi, sigsq_A, sigsq_ABi)

    return var, vi, eni, vi/var, eni/var


def solver_AB(sigsq_A, sigsq_B):
    "Uses f(A) and f(B) to compute solver (mu_hat) variance."
    result = np.concatenate((sigsq_A, sigsq_B),axis=0)
    var_solver = np.mean(result, axis=0) / input_deck.n_eta
    return var_solver


def var_AB(fA, fB, sigsq_A, sigsq_B):
    "Uses f(A) and f(B) to compute total (S_tilde^2), solver (mu_hat), and parametric (S^2) variances."
    var_stand = np.mean((fA-fB)**2, axis=0) / 2
    var_solver = solver_AB(sigsq_A, sigsq_B)
    var_vard = var_stand - var_solver
    return np.array([var_stand, var_vard])


def vi_saltelli(fA, fB, fABi):
    result = fB * (fABi - fA)
    vi_stand = np.mean(result, axis=0)
    vi_vard = vi_stand.copy()
    return np.array([vi_stand, vi_vard])


def eni_jansen(fA, fABi, sigsq_A, sigsq_ABi):
    eni_stand = np.mean((fABi-fA)**2, axis=0) / 2
    var_solver = solver_AB(sigsq_A, sigsq_ABi)
    eni_vard = eni_stand - var_solver
    return np.array([eni_stand, eni_vard])




# Examples
def analytic_ishigami_function(x, a=7, b=0.1):
    """ Test function defined by (Ishigami, Homma 1990). 
    Coefficients (a=7, b=0.1), and (x1, x2, x3) ~U(-pi, pi).
    x is ndarray with shape [3, n_xi]
    y is ndarray with shape [nxi]
    """
    if x.shape[0] != 3:                         # Handles when x is [n_xi,3] rather than [3,n_xi]
        x = x.transpose()
    y = np.sin(x[0]) + a*(np.sin(x[1])**2) + b*(x[2]**4)*np.sin(x[0])
    return y, np.zeros_like(y)


def stochastic_ishigami_function(x, a=7, b=0.1, c=5):
    q, sigma_sq = analytic_ishigami_function(x, a, b)
    eta = np.random.normal(size=(x.shape[0], input_deck.n_eta))
    q_poll = q + c*np.mean(eta, axis=1)
    sigma_sq = (c**2)*np.var(eta, axis=1, ddof=1)
    return q_poll, sigma_sq


def analytic_ishigami_indices(a=7, b=0.1):
    temp1 = (a**2) / 8
    temp2 = b * (np.pi**4) 
    var = 1/2 + temp1 + (temp2**2)/18 + temp2/5
    v1 = ((1 + temp2/5)**2) / 2
    v2 = temp1
    v13 = (temp2**2) * 8 / 225
    eni1 = v1 + v13
    eni2 = v2
    eni3 = v13
    vi = np.array([v1, v2, 0])
    eni = np.array([eni1, eni2, eni3])
    si = vi / var
    ti = eni / var
    return var, vi, eni, si, ti, a/2


# Utilities
def print_error(msg):
    print("ERROR: %s\n" % msg)
    sys.stdout.flush()
    sys.exit()


def read_results(results):
    """ Reads results from `results`.

    Parameters
    ------------
        results_file: str OR dict
            EITHER Name of user-generated .h5 file of results.
            Expected structure below, where there are `k` AB groups
            named using Python indexing, ie. from 0 to k-1 rather than 1 to k.
            |- A/
               |- qoi           N-dim vector of QoIs from f(A), ie. Q-tilde
               |- sigma_sq      N-dim vector of variances from f(A), ie. sigma^2_eta
            |- B/
               |- qoi           N-dim vector of QoIs from f(B), ie. Q-tilde
               |- sigma_sq      N-dim vector of variances from f(B), ie. sigma^2_eta
            |- AB0/
               |- qoi           N-dim vector of QoIs from f(AB^(0)), ie. Q-tilde
               |- sigma_sq      N-dim vector of variances from f(AB^(0)), ie. sigma^2_eta
            |- ...
            OR Dictionary containing results. Expected list of keys:
            fA, fB, fAB, sigsq_A, sigsq_B, sigsq_AB
    """
    k = input_deck.k
    if type(results) is str:
        with h5py.File(results, "r") as f:
            fA = f["A/qoi"][:]
            sigsq_A = f["A/sigma_sq"][:]
            fB = f["B/qoi"][:]
            sigsq_B = f["B/sigma_sq"][:]
            datashape = (k,) + fA.shape
            fAB = np.zeros(datashape)
            sigsq_AB = fAB.copy()
            for i in range(k):
                fAB[i] = f["AB" + str(i) + "/qoi"][:]
                sigsq_AB[i] = f["AB" + str(i) + "/sigma_sq"][:]
    elif type(results) is dict:
        fA = results['fA']
        fB = results['fB']
        fAB = results['fAB']
        sigsq_A = results['sigsq_A']
        sigsq_B = results['sigsq_B']
        sigsq_AB = results['sigsq_AB']
    return fA, fB, fAB, sigsq_A, sigsq_B, sigsq_AB


# Class definition
class InputDeck:
    """
    Storage container for simulation information.
    """
    def __init__(self):
        self.parameters = []
        self.k = 0
        self.n_xi = 0
        self.n_eta = 0
        self.seed = None
        self.matrix = {"A": 0, "B": 0}
        self.rng = None
        self.n_reps = 0
        self.init_seed = None

class Index:
    """ Container for standard (ie total) and var-dec indices. """
    def __init__(self, dshape):
        self.stand = np.empty(shape=dshape)
        self.vard = np.empty(shape=dshape)
    
    def __truediv__(self, other):
        quotient = Index(self.stand.shape)
        quotient.stand = self.stand / other.stand
        quotient.vard = self.vard / other.vard
        return quotient



input_deck = InputDeck()