import jax.numpy as jnp
import numpy as np

np.random.seed(42)


def get_dataset():
    x1 = jnp.array([1.,2.,3.,4.,5.,6.,7.])
    x2 = jnp.array([1.,2.,3.,4.])
    y1 = jnp.array([13.,9.,6.,4.,3.,2.,1.])
    y2 = jnp.array([1.,2.,3.,6.])
    return x1, x2, y1, y2

