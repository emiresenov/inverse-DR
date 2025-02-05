from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, tree_leaves, jacobian, debug

from jaxpi.models import InverseIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from utils import V

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import wandb

class CaseOneField(InverseIVP):
    def __init__(self, config, u_ref, t_star, T_star):
        super().__init__(config)
        self.u_ref = u_ref
        self.t_star = t_star
        self.T_star = T_star

        self.t0 = t_star[0]

        self.u_pred_fn = vmap(vmap(self.u_net, (None, None, 0)), (None, 0, None))


    def u_net(self, params, t, T):
        z = jnp.stack([t, T])
        u = self.state.apply_fn(params, z)
        return u[0], u[1]

    def grad_net(self, params, t, T):
        grads = jacobian(self.u_net, argnums=1)(params, t, T)
        u1_t = grads[0]
        return u1_t

    def r_net(self, params, t, T):
        u1, u2 = self.u_net(params, t, T)
        u1_t = self.grad_net(params, t, T)
        R1 = params['params']['R1']
        C1 = params['params']['C1']
        return u1_t + (u1 - V/u2)/(R1*C1)

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        u1_pred0, u2_pred0 = vmap(self.u_net, (None, None, 0))(params, self.t0, self.T_star)
        R1 = params['params']['R1']
        ic = V/u2_pred0 + V/R1
        ic_loss = jnp.mean((u1_pred0 - ic) ** 2)

        r_pred = vmap(self.r_net, (None, 0, 0))(params, batch[:, 0], batch[:, 1])
        res_loss = jnp.mean((r_pred) ** 2)

        u_pred, _ = self.u_pred_fn(params, self.t_star, self.T_star)
        data_loss = jnp.mean((self.u_ref - jnp.ravel(u_pred)) ** 2)

        loss_dict = {"data": data_loss, "ic": ic_loss, "res": res_loss}
        return loss_dict


    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred, _ = self.u_pred_fn(params, self.t_star, self.T_star)
        error = jnp.linalg.norm(jnp.ravel(u_pred) - u_test) / jnp.linalg.norm(u_test)
        return error


class CaseOneFieldEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params):
        t_star = self.model.t_star
        T_star = self.model.T_star
        u_ref = self.model.u_ref.reshape(T_star.size, t_star.size)
        u_pred, _ = self.model.u_pred_fn(params, t_star, T_star)
        T_mesh, t_mesh = jnp.meshgrid(T_star, t_star, indexing='ij')

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(t_mesh, T_mesh, u_ref, color='red', s=40)
        surf = ax.plot_surface(t_mesh, T_mesh, u_pred.T, cmap='viridis', alpha=0.7, edgecolor='none')

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
