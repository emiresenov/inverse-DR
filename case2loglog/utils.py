import jax.numpy as jnp
import numpy as np

np.random.seed(42)

V = 5.0
R0 = 25.0
R1 = 10.0
C1 = 6.0

R2 = 1.0
C2 = 0.5

t_start = 0.001
t_end = 1000.0
n_samples = 50


def solution(t):
    return jnp.log10(V/R0 + (V/R1)*jnp.exp(-jnp.power(10,t)/(R1*C1)) + (V/R2)*jnp.exp(-jnp.power(10,t)/(R2*C2)))

def get_dataset():
    t = jnp.linspace(jnp.log10(t_start), jnp.log10(t_end), n_samples)
    #noise = np.random.normal(0, 2, len(t))
    #u = solution(t) + noise
    u = solution(t)
    return u, t





#print(get_dataset())


#print(np.linspace(0.1, t_end, n_samples))
#print(np.log10(np.linspace(0.001, t_end, n_samples)))