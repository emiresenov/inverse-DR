from flax import linen as nn
import jax.numpy as jnp

class R0Net(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = jnp.stack([x])
    x = nn.Dense(features=5)(x)
    x = nn.relu(x)
    x = nn.Dense(features=1)(x)
    return x[0]