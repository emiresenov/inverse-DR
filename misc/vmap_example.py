import jax.numpy as jnp
from jax import vmap


from jax import grad, jit



def add(x, y):
    return x * y

# Vectorize the function with batching over the first argument only
batched_add = vmap(add, in_axes=(None, 0))

x = jnp.array([1, 2, 3])
y = jnp.array([4, 5, 6])
result = batched_add(x, y)
print(result)  # Output: [11, 12, 13]