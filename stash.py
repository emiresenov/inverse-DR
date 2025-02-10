def get_dataset2():
    x1 = jnp.array([1.,2.,3.,4.,5.,6.,7.])
    x2 = jnp.array([1.,2.,3.,4.,5.,6.,7.])
    x2 = jnp.array([1.,2.,3.,4.])
    y1 = jnp.array([13.,9.,6.,4.,3.,2.,1.])
    y2 = jnp.array([1.,2.,3.,6.,10.,14.,18.])
    y2 = jnp.array([1.,2.,3.,6.])
    return x1, x2, y1, y2


print(get_dataset())





'''
# STASHED UTILS
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

T_min = jnp.min(Ts)
T_max = jnp.max(Ts)

def scale_T(T):
    return (T - T_min) / (T_max - T_min) 

def rescale_T(T):
    return T * (T_max - T_min) + T_min

def activation_R0(T):
    return L/(a*A*jnp.exp(-(W)/(k * T)))

def solution(t, T):
    return V/activation_R0(T) + (V/R1) * jnp.exp(-t/(R1*C1))

def get_domain():
    return jnp.array([[t_start, t_end], [min(Ts), max(Ts)]])

def get_initial_values():
    t0 = jnp.full(len(Ts), t_start)
    T0 = jnp.array(Ts)
    T0_scaled = scale_T(T0) # IMPORTANT
    return t0, T0_scaled

def get_dataset():
    t = jnp.tile(jnp.linspace(t_start, t_end, n_samples), len(Ts))
    T = jnp.repeat(jnp.array(Ts), n_samples)
    u = solution(t, T)
    T_scaled = scale_T(T) # IMPORTANT
    return t, T_scaled, u, activation_R0(T_scaled)


#print(get_dataset())
#print(get_initial_values())


print(activation_R0(scale_T(Ts)))
'''

