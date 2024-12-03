import scipy.io
import jax.numpy as jnp

U = 1.0
R = 100.0
C = 0.1

t_end = 50.0
n_samples = 15

def solution(t):
    return - t/(R*C)+jnp.log(U/R)

def get_dataset():
    t = jnp.linspace(0.0, t_end, n_samples)
    u = solution(t)
    print(len(t))
    return u, t


