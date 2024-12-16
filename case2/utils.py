from sklearn.preprocessing import normalize
import jax.numpy as jnp
import numpy as np



#np.random.seed(42)

V = 100.0
R_0 = 100.0
R_1 = 10.0
C_1 = 1.0

R_2 = 50.0
C_2 = 10.0

t_end = 5000.0
n_samples = 20000


def solution(t):
    I_01 = V/R_0 + (V/R_1)*jnp.exp(-t/(R_1*C_1))
    I_2 = V/R_2*jnp.exp(-t/(R_2*C_2))
    return I_01 + I_2

def get_dataset():
    t = jnp.linspace(0.0, t_end, n_samples)
    #noise = np.random.normal(0, 2, len(t))
    #u = solution(t) + noise
    u = solution(t)
    return u, t

