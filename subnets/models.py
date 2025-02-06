from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, tree_leaves

from jaxpi.models import InverseIVP, InverseSubnetIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt



class CaseOneField(InverseSubnetIVP):
    def __init__(self, config, x1, x2, y1, y2):
        super().__init__(config)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2


    def u_net(self, params, x1, x2):
        z1 = x1.reshape(-1, 1)
        z2 = x2.reshape(-1, 1)
        u1, u2 = self.state.apply_fn(params, z1, z2)
        return u1[:, 0], u2[:, 0]


    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        u1_pred, u2_pred = self.u_net(params, self.x1, self.x2)
        data1_loss = jnp.mean((self.y1 - u1_pred) ** 2)
        data2_loss = jnp.mean((self.y2 - u2_pred) ** 2)

        loss_dict = {"data1": data1_loss, "data2": data2_loss}

        return loss_dict


    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u1_test, u2_test):
        u1_pred, u2_pred = self.u_net(params, self.x1, self.x2)
        error1 = jnp.linalg.norm(u1_pred - u1_test) / jnp.linalg.norm(u1_test)
        error2 = jnp.linalg.norm(u2_pred - u2_test) / jnp.linalg.norm(u2_test)
        return error1 + error2


class CaseOneFieldEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u1_ref, u2_ref):
        l2_error = self.model.compute_l2_error(params, u1_ref, u2_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params):
        u1_pred, u2_pred = self.model.u_net(params, self.model.x1, self.model.x2)

        fig = plt.figure(figsize=(6, 5))
        plt.scatter(self.model.x1, self.model.y1, s=50, alpha=0.9, c='orange')
        plt.plot(self.model.x1, u1_pred, linewidth=8, c='black')
        self.log_dict["u1_pred"] = fig
        plt.close()

        fig = plt.figure(figsize=(6, 5))
        plt.scatter(self.model.x2, self.model.y2, s=50, alpha=0.9, c='orange')
        plt.plot(self.model.x2, u2_pred, linewidth=8, c='black')
        self.log_dict["u2_pred"] = fig
        plt.close()
    
    def log_inv_params(self, params):
        self.log_dict["R1"] = params['params']['R1'][0]
        self.log_dict["C1"] = params['params']['C1'][0]
        

    def __call__(self, state, batch, y1, y2):
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, y1, y2)

        if self.config.logging.log_preds:
            self.log_preds(state.params)
        
        if self.config.logging.log_inv_params:
            self.log_inv_params(state.params)

        return self.log_dict
    