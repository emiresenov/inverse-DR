import jax.numpy as jnp
import numpy as np

np.random.seed(42)

import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_flatten, tree_unflatten

np.random.seed(42)



def get_dataset():
    x1 = jnp.array([1,2,3,4,5])
    x2 = jnp.array([1,2,3,4,5])
    y1 = jnp.power(x1, 2)
    y2 = jnp.sqrt(x2)
    return x1, x2, y1, y2

print(get_dataset())