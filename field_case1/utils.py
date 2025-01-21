import jax.numpy as jnp
import numpy as np
import jax.tree as tree


np.random.seed(42)

V = 10.0
R0 = 10.0
R1 = 30.0
C1 = 0.007

t_start = 0.0
t_end = 2.0
n_samples = 25

e = 1.602e-19
k = 8.617e-5
W = 0.75
b = 0.01
L = 1
a = 0.5
A = 1

def calc_R0(T):
    return L/(a*A*jnp.exp(-(e*W)/(k*T)))

def solution(t):
    return V/R0 + (V/R1)*jnp.exp(-t/(R1*C1))

def get_dataset():
    t = jnp.linspace(t_start, t_end, n_samples)
    #noise = np.random.normal(0, 2, len(t))
    #u = solution(t) + noise
    u = solution(t)
    return u, t

def update_subnet(params: dict, weights: list):
    leaves, structure = tree.flatten(params)
    assert len(leaves) == len(weights)
    updated_params = tree.unflatten(structure, weights)
    return updated_params



print(calc_R0(293))