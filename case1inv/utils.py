import scipy.io
import jax.numpy as jnp

U = 10.0
R_0 = 25.0
R_1 = 0.5
C_1 = 8.0

t_end = 50.0
n_samples = 100


def solution(t):
    return U/R_0 + (U/R_1)*jnp.exp(-t/(R_1*C_1))

def get_dataset():
    t = jnp.linspace(0.0, t_end, n_samples)
    u = solution(t)
    return u, t
