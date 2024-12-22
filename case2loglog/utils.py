import jax.numpy as jnp
import numpy as np

np.random.seed(42)

V = 10.0
R0 = 25.0
R1 = 0.5
C1 = 8.0

R2 = 0.1
C2 = 0.1

t_start = 0.0001
t_end = 10000.0
n_samples = 80


def solution(t):
    expr1 = V/R0 + (V/R1)*jnp.exp(-jnp.power(10,t)/(R1*C1))
    expr2 = (V/R2)*jnp.exp(-jnp.power(10,t)/(R2*C2))
    return jnp.log10(expr1 + expr2)

def get_dataset():
    t = jnp.linspace(jnp.log10(t_start), jnp.log10(t_end), n_samples)
    u = solution(t)
    return u, t





