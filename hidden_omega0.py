import numpy as np

def get_truth(lb, ub, seed=42):
    rng = np.random.default_rng(seed)

    omega0 = rng.uniform(lb + 1, ub - 1)   # rads/sec
    phi_true = rng.uniform(0, 2*np.pi)

    f0 = omega0 / (2*np.pi)
    T0 = 1 / f0

    return omega0, f0, T0, phi_true