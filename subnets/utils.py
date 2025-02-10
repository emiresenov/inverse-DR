import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten, tree_unflatten

np.random.seed(42)

V = 10.0
R1 = 4.0
C1 = 0.5

t_start = 0.0
t_end = 10.0
n_samples = 15


Ts = jnp.array([0.175, 0.5, 1, 2, 4, 6, 8, 10])


def activation_R0(T):
    return 5/jnp.sqrt(T)

def solution(t, T):
    return V/activation_R0(T) + (V/R1) * jnp.exp(-t/(R1*C1))

def get_domain():
    return jnp.array([[t_start, t_end], [min(Ts), max(Ts)]])

def get_initial_values():
    t0 = jnp.full(len(Ts), t_start)
    T0 = jnp.array(Ts)
    return t0, T0

def get_dataset():
    t = jnp.tile(jnp.linspace(t_start, t_end, n_samples), len(Ts))
    T = jnp.repeat(jnp.array(Ts), n_samples)
    u = solution(t, T)
    #T_scaled = scale_T(T) # IMPORTANT
    return t, T, u, activation_R0(T)


#print(get_dataset())
'''print(get_initial_values())
print(activation_R0(Ts))'''