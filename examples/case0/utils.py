import jax.numpy as jnp

u = 1.0
t_0 = 0.0
t_end = 50.0
r = 1000.0
n_samples = 50
c = 0.01


def solution(t):
    return - t / (r*c) + jnp.log(u/r)


def get_dataset():
    t = jnp.linspace(t_0, t_end, n_samples)
    u = solution(t)
    return t,u


from jax import random
if __name__ == "__main__":
    
    # batch test
    dom = jnp.array([[0., 50.]])
    dim = dom.shape[0]
    batch = random.uniform(
            random.PRNGKey(1234),
            shape=(4, dim),
            minval=dom[:, 0],
            maxval=dom[:, 1],
        )
    print(batch[:, 3])