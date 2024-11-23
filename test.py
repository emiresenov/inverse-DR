import jax.numpy as jnp

d = {
        'R' : jnp.array([800]),
        'C' : jnp.array([0.07])
    }


for i in d:
    print(d[i])
    
    
print(jnp.dot(jnp.array([800]), jnp.array([0.07])))
