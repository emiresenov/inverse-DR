import scipy.io
import jax.numpy as jnp

U = 1.0
T = 10.0
R = 1000.0
C = 0.01

def solution(t):
    return - t / (R * C) + jnp.log(U / R)

def get_dataset():
    t = jnp.linspace(0.0, 50.0, 100)
    u = solution(t)
    return u, t


'''
DEBUG

def gen_traindata():
    t = jnp.linspace(0.0, 10.0, 30)
    u = solution(t)
    return u, t


def get_dataset():
    data = scipy.io.loadmat("data/burgers.mat")
    u_ref = data["usol"]
    t_star = data["t"].flatten()

    return u_ref[:, 0], t_star



print(get_dataset())
print(gen_traindata())

print(get_dataset()[1].shape)
print(gen_traindata()[1].shape)
'''