import jax.numpy as jnp
import numpy as np

#np.random.seed(42)

V = 10.0
R_0 = 25.0
R_1 = 3.5
C_1 = 0.1

R_2 = 8.0
C_2 = 0.5

t_start = 0
t_end = 10.0
n_samples = 100


def solution(t):
    I_01 = V/R_0 + (V/R_1)*jnp.exp(-t/(R_1*C_1))
    I_2 = (V/R_2)*jnp.exp(-t/(R_2*C_2))
    return I_01+I_2

def get_dataset():
    t = jnp.linspace(t_start, t_end, n_samples)
    u = solution(t)
    return u, t

