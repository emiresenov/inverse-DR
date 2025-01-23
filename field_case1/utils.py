import jax.numpy as jnp
import numpy as np
from jax import tree_flatten, tree_unflatten

np.random.seed(42)

V = 10.0
R0 = 10.0
R1 = 30.0
C1 = 0.007

t_start = 0.0
t_end = 2.0
n_samples = 5

e = 1.602e-19
k = 8.617e-5
W = 0.75
b = 0.01
L = 1
a = 0.5
A = 1

Ts = [293.0, 313.0, 333.0]

def get_domain():
    return jnp.array([[t_start, t_end], [min(Ts), max(Ts)]])

def calc_R0(T):
    return L / (a * A * jnp.exp(-(e * W) / (k * T)))

def solution(t, T):
    return V / calc_R0(T) + (V / R1) * jnp.exp(-t / (R1 * C1))

def get_init():
    arr = jnp.array(Ts)
    return jnp.ones_like(arr) * t_start, arr

def get_dataset():
    t_all = []
    u_all = []
    Ts_all = []
    
    t = jnp.linspace(t_start, t_end, n_samples)

    for T in Ts:
        u = solution(t, T)
        Ts_arr = jnp.ones_like(t) * T
        t_all.append(t), u_all.append(u), Ts_all.append(Ts_arr)

    return jnp.concatenate(t_all), jnp.concatenate(u_all), jnp.concatenate(Ts_all)

def update_subnet(params: dict, weights: list):
    leaves, structure = tree_flatten(params)
    assert len(leaves) == len(weights)
    updated_params = tree_unflatten(structure, weights)
    return updated_params


print(get_dataset())
print(get_init())