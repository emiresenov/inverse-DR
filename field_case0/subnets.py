from flax import linen as nn


class ResistanceNet(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1)(x)
    