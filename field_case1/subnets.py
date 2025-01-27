from flax import linen as nn
import jax.numpy as jnp

class R0Net(nn.Module):
    out_dims: int
    @nn.compact
    def __call__(self, x):
        print(f'{x=}')
        print(f'{x.shape=}')
        return nn.Dense(1)(x)
    



print(jnp.ones((1, 3)).shape)
print(jnp.array([222,222,222]).shape)