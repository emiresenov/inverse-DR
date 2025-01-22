import jax.numpy as jnp
import numpy as np
import jax.tree_util as tree

np.random.seed(42)

V = 10.0
R0 = 10.0
R1 = 30.0
C1 = 0.007

t_start = 0.0
t_end = 2.0
n_samples = 5

e = 1.602e-19
k = 8.617e-5
W = 0.75
b = 0.01
L = 1
a = 0.5
A = 1


def calc_R0(T):
    return L / (a * A * jnp.exp(-(e * W) / (k * T)))


def solution(t, T):
    return V / calc_R0(T) + (V / R1) * jnp.exp(-t / (R1 * C1))


def get_dataset(T):
    t = jnp.linspace(t_start, t_end, n_samples)
    u = solution(t, T)
    Ts = jnp.ones((n_samples,)) * T
    return jnp.array([u, t, Ts])

datasets = []

Temperatures = [293.0, 313.0, 333.0]

for T in Temperatures:
    dataset = get_dataset(T)
    datasets.append(dataset)

arr = jnp.stack(datasets, axis=0)

print(arr)

print(f'{arr[1][0]=}')