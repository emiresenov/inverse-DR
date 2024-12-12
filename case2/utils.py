import scipy.io
import jax.numpy as jnp
import numpy as np

np.random.seed(42)

U = 5.0
R_0 = 25.0
R_1 = 3.0
C_1 = 0.05 #TODO: Sanity check: kolla på miniräknaren att I(t=0) uppfyller begynnelsevillkoret

R_2 = 5.0
C_2 = 0.45

t_end = 10.0
n_samples = 50


def solution(t):
    I_01 = U/R_0 + (U/R_1)*jnp.exp(-t/(R_1*C_1))
    I_2 = U/R_2*jnp.exp(-t/(R_2*C_2))
    return I_01 + I_2

def get_dataset():
    t = jnp.linspace(0.0, t_end, n_samples)
    noise = np.random.normal(0, 2, len(t))
    u = solution(t) + noise
    return u, t
