import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten, tree_unflatten

np.random.seed(42)

V = 10.0
R1 = 4.0
C1 = 0.5

t_start = 0.0
t_end = 10.0
n_samples = 25

k = 8.617e-5
W = 0.75
b = 0.01
L = 1e-6
a = 2000
A = 200

Ts = [293.0, 313.0, 333.0] # 4-10 serier, 20-100 grader

def get_domain():
    return jnp.array([[t_start, t_end], [min(Ts), max(Ts)]])

def calc_R0(T):
    return L / (a * A * jnp.exp(-(W) / (k * T)))

def solution(t, R0):
    return V / R0 + (V / R1) * jnp.exp(-t / (R1 * C1))

def get_initial_values():
    arr = jnp.array(Ts)
    return jnp.ones_like(arr) * t_start, arr

def get_dataset():
    t_all = []
    u_all = []
    T_all = []
    R0_all = []
    t = jnp.linspace(t_start, t_end, n_samples)
    for T in Ts:
        R0 = calc_R0(T)
        u = solution(t, R0)
        Ts_arr = jnp.ones_like(t) * T
        t_all.append(t), u_all.append(u), T_all.append(Ts_arr), R0_all.append(R0)
    return jnp.concatenate(u_all), jnp.array(R0_all), jnp.concatenate(t_all) , jnp.concatenate(T_all)

def update_subnet(params: dict, weights: list):
    leaves, structure = tree_flatten(params)
    assert len(leaves) == len(weights)
    updated_params = tree_unflatten(structure, weights)
    return updated_params

dataset = get_dataset()
print(dataset[0])
print(dataset[1])
print(dataset[2])
print(dataset[3])
#print(get_domain()[:, 0])
#for T in Ts:
    #print(calc_R0(T))


