from flax import linen as nn
from flax.training import train_state
from jax import random, jit, value_and_grad, vmap
import jax.numpy as jnp
import optax
import jax

# Define an MLP that expects an input of shape (1,) and returns an output of shape (1,).
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        # x is assumed to be an array of shape (1,)
        x = nn.Dense(features=5)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x

# Extend the training state (no extra fields needed here).
class TrainState(train_state.TrainState):
    pass

# Mean squared error loss.
# Here, we assume that batch_x and batch_y have shapes (batch,) — i.e. each element is a scalar.
# We use vmap to apply the model to each scalar by wrapping it in a (1,) array.
def mse_loss(params, apply_fn, batch_x, batch_y):
    # For each scalar x in batch_x, wrap it as an array of shape (1,) before passing it to the model.
    preds = vmap(lambda x: apply_fn(params, jnp.array([x])))(batch_x)  # shape (batch, 1)
    preds = preds.squeeze(-1)  # convert to shape (batch,)
    loss = jnp.mean((preds - batch_y) ** 2)
    return loss

# A single training step (jitted).
@jit
def train_step(state, x, y):
    loss_fn = lambda params: mse_loss(params, state.apply_fn, x, y)
    loss, grads = value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

# Training loop.
def train_model(model, x, y, num_epochs=100, learning_rate=0.01):
    key = random.PRNGKey(0)
    # Use a dummy input that matches what the model expects for a single example.
    dummy_input = jnp.ones((1,))   # shape (1,)
    params = model.init(key, dummy_input)
    tx = optax.adam(learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    for epoch in range(num_epochs):
        state, loss = train_step(state, x, y)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    return state

# ===== Training Phase =====
# Now the training data are one-dimensional arrays (shape (3,)) rather than two-dimensional arrays.
x_train = jnp.array([0., 5., 10.])
y_train = jnp.array([20.5, 3.5, 0.4])

# Instantiate and train the model.
model = MLP()
state = train_model(model, x_train, y_train, num_epochs=4000, learning_rate=0.01)

# ===== Inference with vmap =====
# Define a helper function that applies the model to a single scalar input.
def apply_single(params, x_scalar):
    # Wrap the scalar into an array of shape (1,)
    x_wrapped = jnp.array([x_scalar])
    # Apply the model; output will be an array of shape (1,)
    y_out = model.apply(params, x_wrapped)
    return y_out[0]  # Return the scalar value

# Use vmap to vectorize our single-input function.
vectorized_apply = vmap(lambda x: apply_single(state.params, x), in_axes=0)
outputs = vectorized_apply(jnp.array([0., 5., 10.]))

print("Vectorized outputs:\n", outputs)



#y = jnp.array([1., 2., 3.])
#y = jnp.array([[1.], [2.], [3.]])  # Ensure y has shape (batch, features)
#print(vmap(model.apply, (None, 0))(params, y))
#print(params)
#print(jnp.stack(jnp.array([[1., 2., 3.]])))


#from flax.traverse_util import flatten_dict
#import jax.tree_util as jtu











# -----------------------------
#  UPDATE PARAMS TEST
# -----------------------------

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