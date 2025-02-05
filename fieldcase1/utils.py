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

Ts = [293.0, 313.0, 333.0, 353.0] # 4-10 serier, 293-373 Kelvin

def get_domain():
    return jnp.array([[t_start, t_end], [min(Ts), max(Ts)]])

def activation_R0(T):
    return L/(a*A*jnp.exp(-(W)/(k * T)))

def solution(t, T):
    return V/activation_R0(T) + (V/R1) * jnp.exp(-t/(R1*C1))

def get_dataset():
    t = jnp.linspace(t_start, t_end, n_samples)
    u_all = []
    for T in Ts:
        u = solution(t, T)
        u_all.append(u)
    return jnp.concatenate(u_all), t, jnp.array(Ts)



'''dataset = get_dataset()
print(dataset[0])
print(dataset[1])
print(dataset[2])'''
#print(get_domain())
#print(get_initial_values())


