import scipy.io
import jax.numpy as jnp
import numpy as np

np.random.seed(42)

U = 1.0
R = 100.0
C = 0.1

t_end = 50.0
n_samples = 40

def solution(t):
    return - t/(R*C)+jnp.log(U/R)

def get_dataset():
    t = jnp.linspace(0.0, t_end, n_samples)
    noise = np.random.normal(0, 1, len(t))
    u = solution(t) + noise
    #u = solution(t)
    return u, t


