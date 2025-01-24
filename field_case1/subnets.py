from flax import linen as nn


class R0Net(nn.Module):
    out_dims: int
    @nn.compact
    def __call__(self, x):
        print(f'{x=}')
        return nn.Dense(1)(x)
    