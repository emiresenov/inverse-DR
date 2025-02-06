from flax import linen as nn
from flax.training import train_state
from jax import random, jit, value_and_grad, vmap
import jax.numpy as jnp
import optax


x1 = jnp.array([1.,2.,3.,4.,5.,6.,7.])

print(jnp.stack([x1]))
