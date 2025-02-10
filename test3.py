from flax import linen as nn
from flax.training import train_state
from jax import random, jit, value_and_grad, vmap
import jax.numpy as jnp
import optax


#x1 = jnp.array([1.,2.,3.,4.,5.,6.,7.])

#print(jnp.stack([x1, x1], axis=1))




V = 10
R1 = 3
R0 = jnp.array([1,2,3,4])
ic = jnp.array([5,4,3,2])


expr = ic - (V/R0 + V/R1) 
print(expr)