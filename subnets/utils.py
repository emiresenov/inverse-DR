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

#Ts = [100.,200.,300.,400.] # 4-10 serier, 293-373 Kelvin

Ts = jnp.array([
    293.0, 
     300.0, 
     305.0, 
     313.0, 
     315.0, 
     320.0, 
     333.0, 
     335.0, 
     340.0, 
     345.0, 
     353.0
]) # 4-10 serier, 293-373 Kelvin

Ts = (Ts - jnp.min(Ts)) / (jnp.max(Ts) - jnp.min(Ts)) # Normalize

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
    t = jnp.linspace(t_start, t_end, n_samples)
    #x1 = jnp.tile(t, len(Ts))
    x1 = t
    x2 = jnp.array(Ts)
    y1_all = []
    y1_all.append(solution(t, Ts[0])) # TODO: REMOVE
    '''for T in Ts:
        y1 = solution(t, T)
        y1_all.append(y1)'''

     # TODO: Remove. Adding this just to make sure that subnet r0 is functioning properly
    y2_dummy_array = x2 * 3

    return x1, x2, jnp.concatenate(y1_all), y2_dummy_array

'''def get_dataset2():
    x1 = jnp.array([1.,2.,3.,4.,5.,6.,7.])
    x2 = jnp.array([1.,2.,3.,4.,5.,6.,7.])
    x2 = jnp.array([1.,2.,3.,4.])
    y1 = jnp.array([13.,9.,6.,4.,3.,2.,1.])
    y2 = jnp.array([1.,2.,3.,6.,10.,14.,18.])
    y2 = jnp.array([1.,2.,3.,6.])
    return x1, x2, y1, y2'''

#print(get_dataset())
#print(get_dataset2())