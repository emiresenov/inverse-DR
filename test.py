from flax import linen as nn
from jax import random, vmap
import jax.numpy as jnp

class MLP(nn.Module):
  out_dims: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(1)(x)
    return x

model = MLP(out_dims=1)

x = jnp.empty((3,10))
params = model.init(random.key(42), x)
#y = model.apply(params, x)   
#print(y)
mapper = vmap(model.apply, (None, 0))
print(mapper(params, x))
print(x)