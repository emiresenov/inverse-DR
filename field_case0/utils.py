import scipy.io
import jax.numpy as jnp
import numpy as np
from jax import tree_flatten, tree_unflatten


#np.random.seed(42)

V = 10.0
R = 10.0
C = 0.1

t_end = 7.0
n_samples = 10


def r(t):
    return -0.5*t + R

def c(t):
    return t/100 + C

def solution(t):
    return (V/r(t))*jnp.exp(-t/(r(t)*c(t)))

def get_dataset():
    t = jnp.linspace(0.0, t_end, n_samples)
    #noise = np.random.normal(0, 0.3, len(t))
    #u = solution(t) + noise
    u = solution(t)
    return u, t


def update_subnet(params: dict, weights: list):
    leaves, structure = tree_flatten(params)
    assert len(leaves) == len(weights)
    updated_params = tree_unflatten(structure, weights)
    return updated_params