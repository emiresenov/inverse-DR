import scipy.io
import jax.numpy as jnp
import numpy as np

#np.random.seed(42)

V = 10.0
R = 10.0
C = 0.1

t_end = 7.0
n_samples = 10

def solution(t):
    return (V/R)*jnp.exp(-t/(R*C))

def get_dataset():
    t = jnp.linspace(0.0, t_end, n_samples)
    #noise = np.random.normal(0, 0.3, len(t))
    #u = solution(t) + noise
    u = solution(t)
    return u, t


u, t = get_dataset()
print(t[0])