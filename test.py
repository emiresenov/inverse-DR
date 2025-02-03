from flax import linen as nn
from flax.training import train_state
from jax import random, jit, value_and_grad, vmap
import jax.numpy as jnp
import optax


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = jnp.stack([x])
        x = nn.Dense(features=5)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x[0]

class TrainState(train_state.TrainState):
    pass

def mse_loss(params, apply_fn, x, y):
    preds = vmap(lambda x: apply_fn(params, x))(x)
    return jnp.mean((preds - y) ** 2)

@jit
def train_step(state, x, y):
    loss_fn = lambda params: mse_loss(params, state.apply_fn, x, y)
    loss, grads = value_and_grad(loss_fn)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

def train_model(model, x, y, num_epochs=100, learning_rate=0.01):
    params = model.init(random.PRNGKey(0), jnp.array([1.]))
    tx = optax.adam(learning_rate)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    for epoch in range(num_epochs):
        state, loss = train_step(state, x, y)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    return state


x = jnp.array([0., 5., 10.])
y = jnp.array([20.5, 3.5, 0.4])

model = MLP()
state = train_model(model, x, y, num_epochs=4000, learning_rate=0.01)

vectorized_apply = vmap(model.apply, (None, 0))
outputs = vectorized_apply(state.params, x)

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