from flax import linen as nn
from jax import random, vmap
import jax.numpy as jnp
from flax.traverse_util import flatten_dict
import jax.tree_util as jtu

class MLP(nn.Module):
  out_dims: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(1)(x)
    return x

model = MLP(out_dims=1)

x = jnp.empty((1,10))
params = model.init(random.key(42), x)

param_values = jtu.tree_leaves(params)

print(type(params))
print(type(param_values))
print(param_values)  # List of arrays

param_values[0] = [0.5]
param_values[1] = param_values[1].at[0].set(0.)

leaves, structure = jtu.tree_flatten(params)

assert len(leaves) == len(param_values)

params = jtu.tree_unflatten(structure, param_values)

print(params)