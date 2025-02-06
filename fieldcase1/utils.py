import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten, tree_unflatten

np.random.seed(42)

V = 10.0
R1 = 4.0
C1 = 0.5

t_start = 0.0
t_end = 10.0
n_samples = 1

k = 8.617e-5
W = 0.75
b = 0.01
L = 1e-6
a = 2000
A = 200

Ts = [293.0, 300.0, 305.0, 313.0, 315.0, 320.0, 333.0, 335.0, 340.0, 345.0, 353.0] # 4-10 serier, 293-373 Kelvin


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
    times = jnp.tile(t, len(Ts))
    T_array = jnp.array(Ts)
    temperatures = jnp.repeat(T_array, n_samples)
    x = jnp.column_stack((times, temperatures))
    y = solution(times, temperatures)
    return x, y

def y2_ref():
    T_array = jnp.array(Ts)
    temperatures = jnp.repeat(T_array, n_samples)
    R0 = activation_R0(temperatures)
    return R0

#print(get_dataset())

#print(get_initial_values())
#print(y2_ref())