import jax.numpy as jnp
import numpy as np

#np.random.seed(42)

V = 5.0
R_0 = 25.0
R_1 = 5.0
C_1 = 0.05

R_2 = 8.0
C_2 = 0.45

'''t_start = 0.0001
t_end = 10000.0
n_samples = 50'''

t_start = 0
t_end = 6.0
n_samples = 100


'''def solution(t):
    I_01 = V/R_0 + (V/R_1)*jnp.exp(-jnp.power(10,t)/(R_1*C_1))
    I_2 = (V/R_2)*jnp.exp(-jnp.power(10,t)/(R_2*C_2))
    return jnp.log10(I_01+I_2)

def get_dataset():
    t = jnp.linspace(jnp.log10(t_start), jnp.log10(t_end), n_samples)
    u = solution(t)
    return u, t'''
    
def solution(t):
    I_01 = V/R_0 + (V/R_1)*jnp.exp(-t/(R_1*C_1))
    I_2 = (V/R_2)*jnp.exp(-t/(R_2*C_2))
    return I_01+I_2

def get_dataset():
    t = jnp.linspace(t_start, t_end, n_samples)
    u = solution(t)
    return u, t

