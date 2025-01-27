from flax import linen as nn
import jax.numpy as jnp

class R0Net(nn.Module):
  @nn.compact
  def __call__(self, x):
    return nn.Dense(1)(jnp.stack([x]))