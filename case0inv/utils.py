import scipy.io
import jax.numpy as jnp

U = 1.0
T = 10.0
R = 1000.0
C = 0.01

def solution(t):
    return - t / (R * C) + jnp.log(U / R)

def get_dataset():
    t = jnp.linspace(0.0, 50.0, 30)
    u = solution(t)
    return u, t
