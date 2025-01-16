import jax.numpy as jnp
import numpy as np

#np.random.seed(42)

V = 10.0
R0 = 25.0
R1 = 1.0
C1 = 0.1

R2 = 4.5
C2 = 1.0

R3 = 25.0
C3 = 25.0

t_start = 0
t_end = 4000.0
n_samples = 4000


def solution(t):
    I_01 = V/R0 + (V/R1)*jnp.exp(-t/(R1*C1))
    I_2 = (V/R2)*jnp.exp(-t/(R2*C2))
    I_3 = (V/R3)*jnp.exp(-t/(R3*C3))
    return I_01+I_2+I_3

def get_dataset():
    t = jnp.linspace(t_start, t_end, n_samples)
    u = solution(t)
    return u, t

