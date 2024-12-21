import jax.numpy as jnp
import numpy as np

np.random.seed(42)

V = 10.0
R_0 = 25.0
R_1 = 0.5
C_1 = 8.0

t_end = 100.0
n_samples = 80


def solution(t):
    return jnp.log10(V/R_0+(V/R_1)*jnp.exp(-jnp.power(t,10)/(R_1*C_1)))

def get_dataset():
    t = jnp.linspace(0.1, t_end, n_samples)
    #noise = np.random.normal(0, 2, len(t))
    #u = solution(t) + noise
    u = solution(t)
    
    return u, jnp.linspace(0.1, jnp.log10(t_end), n_samples)





#print(get_dataset())


#print(np.linspace(0.1, t_end, n_samples))
#print(np.log10(np.linspace(0.001, t_end, n_samples)))