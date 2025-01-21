from flax import linen as nn


class R0Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1)(x)
    