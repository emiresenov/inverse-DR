import jax.numpy as jnp
import numpy as np

np.random.seed(42)

V = 10.0
R1 = 4.0
C1 = 0.5

t_start = 0.0
t_end = 3
n_samples = 20

a = 1e-13
W = 0.75
k = 8e-5

Ts = jnp.linspace(293.15, 323.15, 10)
t_scale = 330.15

p_noise = 0.4

R1_const = 0.5

def activation_R0(T):
    return a*jnp.exp(W/(k*T))

def activation_R1(T):
    return R1_const*activation_R0(T)

def solution(t, T):
    R1 = activation_R1(T)
    return jnp.log(V/activation_R0(T) + (V/R1) * jnp.exp(-t/(R1*C1)))

def get_domain():
    return jnp.array([[t_start, t_end], [min(Ts)/t_scale, max(Ts)/t_scale]])

def get_ic_dom():
    t0 = jnp.full(len(Ts), t_start)
    T0 = jnp.array(Ts) / t_scale
    return t0, T0

def get_dataset():
    t = jnp.tile(jnp.linspace(t_start, t_end, n_samples), len(Ts))
    T = jnp.repeat(jnp.array(Ts), n_samples)
    u1 = solution(t, T)
    u2 = activation_R0(T)
    noise_level = p_noise / (jnp.abs(u1) + 1e-3)
    noise = noise_level * np.random.normal(size=u1.shape)
    u1 = u1 + noise
    return t, T / t_scale, u1, u2

def get_ic_ref():
    t0s = jnp.repeat(t_start, len(Ts))
    return solution(t0s, Ts)

