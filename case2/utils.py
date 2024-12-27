import jax.numpy as jnp
import numpy as np

#np.random.seed(42)

V = 10.0
R0 = 25.0
R1 = 3.5
C1 = 0.1

R2 = 8.0
C2 = 0.5

t_start = 0
t_end = 10.0
n_samples = 25


def solution(t):
    I_01 = V/R0 + (V/R1)*jnp.exp(-t/(R1*C1))
    I_2 = (V/R2)*jnp.exp(-t/(R2*C2))
    return I_01+I_2

def get_dataset():
    t = jnp.linspace(t_start, t_end, n_samples)
    u = solution(t)
    return u, t

