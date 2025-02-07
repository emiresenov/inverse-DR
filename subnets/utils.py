import jax.numpy as jnp
import numpy as np

np.random.seed(42)

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

Ts = jnp.array([
     293.0, 
     305.0, 
     313.0, 
     315.0, 
     333.0, 
     335.0,
     345.0, 
     353.0
]) # 4-10 series, 293-373 Kelvin

# Normalize temperatures (important!)
Ts = (Ts - jnp.min(Ts)) / (jnp.max(Ts) - jnp.min(Ts)) 

def activation_R0(T):
    return L/(a*A*jnp.exp(-(W)/(k * T)))

def solution(t, T):
    return V/activation_R0(T) + (V/R1) * jnp.exp(-t/(R1*C1))

def get_domain():
    return jnp.array([[t_start, t_end], [min(Ts), max(Ts)]])

def get_initial_values():
    times = jnp.full(len(Ts), t_start)
    temperatures = jnp.array(Ts)
    return jnp.column_stack((times, temperatures))

def get_dataset():
    t_ref = jnp.linspace(t_start, t_end, n_samples)
    '''t = jnp.linspace(t_start, t_end, n_samples)
    t_ref = jnp.tile(t, len(Ts))'''
    T_ref = jnp.array(Ts)
    u1_ref = []
    u1_ref.append(solution(t_ref, Ts[0])) # TODO: REMOVE
    '''for T in Ts:
        u1 = solution(t, T)
        u1_ref.append(u1)'''

     # TODO: Remove. Adding this just to make sure that subnet r0 is functioning properly
    u2_dummy_array = T_ref * 3

    return t_ref, T_ref, jnp.concatenate(u1_ref), u2_dummy_array
