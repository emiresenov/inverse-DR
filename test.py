from flax import linen as nn
from flax.training import train_state
from jax import random, jit, value_and_grad
import jax.numpy as jnp
import optax
import jax


# Define the MLP model.
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=5)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x


# Extend the train state (no extra fields needed).
class TrainState(train_state.TrainState):
    pass


# Use 'apply_fn' as the argument name instead of 'model'
def mse_loss(params, apply_fn, batch_x, batch_y):
    preds = apply_fn(params, batch_x)
    loss = jnp.mean((preds - batch_y) ** 2)
    return loss


# Jitted training step.
@jax.jit
def train_step(state, x, y):
    loss_fn = lambda params: mse_loss(params, state.apply_fn, x, y)
    loss, grads = value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss


# Training loop.
def train_model(model, x, y, num_epochs=100, learning_rate=0.01):
    key = random.PRNGKey(0)
    # Dummy input to initialize the model.
    dummy_input = jnp.ones((1, x.shape[1]))
    params = model.init(key, dummy_input)
    tx = optax.adam(learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    for epoch in range(num_epochs):
        state, loss = train_step(state, x, y)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

    return state


# Instantiate the model.
model = MLP()

# Prepare data: three samples, each with one feature.
x = jnp.array([[0.], [5.], [10.]])
y = jnp.array([[20.5], [3.5], [0.4]])

# Train the model.
state = train_model(model, x, y, num_epochs=15000, learning_rate=0.01)

# Apply the trained model.
output = model.apply(state.params, x)
print("Output:\n", output)



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