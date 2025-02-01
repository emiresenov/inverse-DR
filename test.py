from flax import linen as nn
from jax import random, vmap, tree_flatten, tree_unflatten, tree_leaves
import jax.numpy as jnp                                   


class MLP(nn.Module):
  out_dims: int

  @nn.compact
  def __call__(self, x):
    return nn.Dense(1)(jnp.stack([x]))

model = MLP(out_dims=1)

x = jnp.array([1.])
params = model.init(random.key(42), x)



#y = jnp.array([1., 2., 3.])
#y = jnp.array([[1.], [2.], [3.]])  # Ensure y has shape (batch, features)
#print(vmap(model.apply, (None, 0))(params, y))
#print(params)
#print(jnp.stack(jnp.array([[1., 2., 3.]])))


#from flax.traverse_util import flatten_dict
#import jax.tree_util as jtu

'''param_values = tree_leaves(params)

print(type(params))
print(type(param_values))
print(param_values)  # List of arrays

param_values[0] = [0.5]
param_values[1] = param_values[1].at[0].set(0.)

leaves, structure = tree_flatten(params)

assert len(leaves) == len(param_values)

params = tree_unflatten(structure, param_values)

print(params)'''