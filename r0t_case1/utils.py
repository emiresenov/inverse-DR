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


a = 1e-13
W = 0.75
k = 8e-5

Ts = jnp.linspace(293.15, 320.15, 10)
t_scale = 330.15

p_noise = 0.5

def activation_R0(T):
    return a*jnp.exp(W/(k*T))

def solution(t, T):
    return V/activation_R0(T) + (V/R1) * jnp.exp(-t/(R1*C1))

def get_domain():
    return jnp.array([[t_start, t_end], [min(Ts)/t_scale, max(Ts)/t_scale]]) # TODO: RM

def get_initial_values():
    t0 = jnp.full(len(Ts), t_start)
    T0 = jnp.array(Ts) / t_scale # TODO: RM
    return t0, T0

def get_dataset():
    t = jnp.tile(jnp.linspace(t_start, t_end, n_samples), len(Ts))
    T = jnp.repeat(jnp.array(Ts), n_samples)
    u1 = solution(t, T)
    #noise = np.random.normal(loc=0.0, scale=p_noise*jnp.mean(u1), size=u1.shape)
    #noise = np.random.normal(loc=0.0, scale=p_noise, size=u1.shape)
    #u1 = u1 + noise
    u2 = activation_R0(T)
    return t, T / t_scale, u1, u2 # TODO:RM

def get_ic():
    t0s = jnp.repeat(t_start, len(Ts))
    return solution(t0s, Ts)



'''print(get_dataset())
print()
print(get_initial_values())
print()
print(get_domain())'''


'''t = jnp.tile(jnp.linspace(t_start, t_end, n_samples), len(Ts))
T = jnp.repeat(jnp.array(Ts), n_samples)
u1 = solution(t, T)
print(jnp.log(u1))'''












'''T_scale = 8.0

Ts = jnp.linspace(0, 14, 10)

p_noise = 0.5

def activation_R0(T):
    return 20*jnp.exp(-0.5*T)

def solution(t, T):
    return V/activation_R0(T) + (V/R1) * jnp.exp(-t/(R1*C1))

def get_domain():
    return jnp.array([[t_start, t_end], [min(Ts)/T_scale, max(Ts)/T_scale]]) # TODO: RM

def get_initial_values():
    t0 = jnp.full(len(Ts), t_start)
    T0 = jnp.array(Ts)/T_scale # TODO: RM
    return t0, T0

def get_dataset():
    t = jnp.tile(jnp.linspace(t_start, t_end, n_samples), len(Ts))
    T = jnp.repeat(jnp.array(Ts), n_samples)
    u1 = solution(t, T)
    #noise = np.random.normal(loc=0.0, scale=p_noise*jnp.mean(u1), size=u1.shape)
    #noise = np.random.normal(loc=0.0, scale=p_noise, size=u1.shape)
    #u1 = u1 + noise
    u2 = activation_R0(T)
    return t, T/T_scale, u1, u2 # TODO:RM

def get_ic():
    t0s = jnp.zeros(len(Ts)) # TODO: CHANGE TO REPEAT t_start instead of assuming t_start = 0 (jnp.zeros)
    return solution(t0s, Ts)
'''
