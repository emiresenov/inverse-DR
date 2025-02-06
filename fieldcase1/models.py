from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, tree_leaves, jacobian, debug

from jaxpi.models import InverseIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from utils import V, get_initial_values, y2_ref

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import wandb

class CaseOneField(InverseIVP):
    def __init__(self, config, x_ref, y1_ref):
        super().__init__(config)
        self.x_ref = x_ref
        self.y1_ref = y1_ref
        self.y2_ref = y2_ref()
        self.x0 = get_initial_values()
        self.y_pred_fn = vmap(self.y_net, (None, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0))


    def y_net(self, params, x):
        return self.state.apply_fn(params, x)
    
    def y1_net(self, params, x):
        return self.y_net(params, x)[0]

    def grad_net(self, params, x):
        grad_val = grad(self.y1_net, argnums=1)(params, x)
        return grad_val[0]

    def r_net(self, params, x):
        y1,y2 = self.y_net(params, x)
        y1_t = self.grad_net(params, x)
        R1 = params['params']['R1']
        C1 = params['params']['C1']
        return y1_t + (y1-V/y2)/(R1*C1)

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        y0 = self.y_pred_fn(params, self.x0)
        y1_t0 = y0[:, 0] #I
        y2_t0 = y0[:, 1] #R0
        R1 = params['params']['R1']
        ic = V/y2_t0 + V/R1
        ic_loss = jnp.mean((y1_t0 - ic) ** 2)

        r_pred = self.r_pred_fn(params, batch)
        res_loss = jnp.mean((r_pred) ** 2)

        y_pred = self.y_pred_fn(params, self.x_ref)
        y1_pred = y_pred[:, 0]
        data_loss = jnp.mean((y1_pred - self.y1_ref) ** 2)

        y2_pred = y_pred[:, 1]
        y2_loss = jnp.mean((y2_pred - self.y2_ref)**2)

        loss_dict = {"data": data_loss, "ic": ic_loss, "res": res_loss, "y2": y2_loss}
        return loss_dict


    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, y1_test):
        y1_pred = self.y_pred_fn(params, self.x_ref)[:, 0]
        error = jnp.linalg.norm(jnp.ravel(y1_pred) - y1_test) / jnp.linalg.norm(y1_test)
        return error


class CaseOneFieldEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params):
        temperaturer = 11
        samples = 50
        x = self.model.x_ref
        x1 = x[:,0].reshape(temperaturer, samples) # HARDCODED
        x2 = x[:,1].reshape(temperaturer, samples) # HARDCODED
        y1_pred = self.model.y_pred_fn(params, self.model.x_ref)[:,0]
        y1_pred = y1_pred.reshape(temperaturer, samples) # HARDCODED
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[:, 0], x[:, 1], self.model.y1_ref, color='red', s=40)
        surf = ax.plot_surface(x1, x2, y1_pred, cmap='viridis', alpha=0.7, edgecolor='none')
        fig.colorbar(surf, shrink=0.5, aspect=5, label='u')
        self.log_dict["u_pred"] = wandb.Image(fig)
        plt.close()
    
    def log_inv_params(self, params):
        self.log_dict["R1"] = params['params']['R1'][0]
        self.log_dict["C1"] = params['params']['C1'][0]
        

    def __call__(self, state, batch, u_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params)
        
        if self.config.logging.log_inv_params:
            self.log_inv_params(state.params)

        return self.log_dict
