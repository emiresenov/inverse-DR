import jax.numpy as jnp
import numpy as np

np.random.seed(42)

'''V = 10.0
R_0 = 25.0
R_1 = 0.5
C_1 = 8.0

t_end = 50.0
n_samples = 40'''

V = 1000.0
R_0 = 5000.0
R_1 = 15.0
C_1 = 3000.0

t_end = 1000000.0
n_samples = 1000


def solution(t):
    return V/R_0 + (V/R_1)*jnp.exp(-t/(R_1*C_1))

def get_dataset():
    t = jnp.linspace(0.0, t_end, n_samples)
    #noise = np.random.normal(0, 2, len(t))
    #u = solution(t) + noise
    u = solution(t)

    # Normalization
    u_normalized = u / jnp.max(u)
    t_normalized = t / jnp.max(t)
    return u_normalized, t_normalized
    
    #return u, t


def get_umax():
    t = jnp.linspace(0.0, t_end, n_samples)
    return jnp.max(solution(t))


print(get_dataset())