from flax import linen as nn
from flax.training import train_state
from jax import random, jit, value_and_grad, vmap
import jax.numpy as jnp
import optax

# MLP with 2 outputs
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x will be shape (2,)
        x = nn.Dense(features=5)(x)
        x = nn.relu(x)
        x = nn.Dense(features=2)(x)  # output shape (2,)
        return x

class TrainState(train_state.TrainState):
    pass

# Only use the FIRST output in the MSE
def mse_loss(params, apply_fn, x, y):
    # preds will have shape (batch_size, 2)
    preds = vmap(lambda single_x: apply_fn(params, single_x))(x)
    # Compare only the first output to y
    return jnp.mean((preds[:, 0] - y) ** 2)

@jit
def train_step(state, x, y):
    loss_fn = lambda params: mse_loss(params, state.apply_fn, x, y)
    loss, grads = value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

def train_model(model, x, y, num_epochs=100, learning_rate=0.01):
    # Initialize the model with an example of the correct input shape (2,)
    params = model.init(random.PRNGKey(0), jnp.array([0., 0.]))
    tx = optax.adam(learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    for epoch in range(num_epochs):
        state, loss = train_step(state, x, y)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    return state

# Sample 2D input data (x)
x = jnp.array([
    [0., 1.],
    [1., 2.],
    [2., 3.],
    [3., 4.]
])

# Single-column (or 1D) target data (y)
# Note: shape (4,) or (4, 1) is fine; here we use shape (4,)
y = jnp.array([0., 1., 1., 2.])

model = MLP()
state = train_model(model, x, y, num_epochs=200, learning_rate=0.01)

# Vectorized prediction
vectorized_apply = vmap(model.apply, (None, 0))
outputs = vectorized_apply(state.params, x)

print("All model outputs (2 neurons per input):\n", outputs)
print("First output vs. target (used in loss):")
for o, tgt in zip(outputs[:, 0], y):
    print(f"Prediction = {o:.3f}, Target = {tgt:.3f}")
