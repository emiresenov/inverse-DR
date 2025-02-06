import jax.numpy as jnp
import numpy as np

np.random.seed(42)

V = 10.0
R0 = 2.0
R1 = 4.0
C1 = 0.5

t_start = 0.0
t_end = 10.0
n_samples = 25


def solution(t):
    return V/R0 + (V/R1)*jnp.exp(-t/(R1*C1))

def get_dataset():
    t = jnp.linspace(t_start, t_end, n_samples)
    #noise = np.random.normal(0, 2, len(t))
    #u = solution(t) + noise
    u = solution(t)
    return u, t
