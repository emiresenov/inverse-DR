from flax import linen as nn


class Resistance(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1)(x)
    
    
class Capacitance(nn.Module):
    @nn.compact
    def __call__(self, x):
        return nn.Dense(1)(x)