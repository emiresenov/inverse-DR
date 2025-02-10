from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, tree_leaves

from jaxpi.models import InverseIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import wandb

from utils import V, get_initial_values


class CaseOneField(InverseIVP):
    def __init__(self, config, t_ref, T_ref, u1_ref, u2_ref):
        super().__init__(config)
        self.t_ref = t_ref
        self.T_ref = T_ref
        self.u1_ref = u1_ref
        self.u2_ref = u2_ref

        self.u_pred_fn = vmap(self.u_net, (None, 0, 0))
        #self.r_pred_fn = vmap(vmap(self.r_net, (None, 0, None)), (None, None, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0))
        

    def u_net(self, params, t, T):
        z = jnp.stack([t, T])
        u1, u2 = self.state.apply_fn(params, z)
        return u1[0], u2[0]
    
    def u1_net(self, params, t, T):
        u1, _ = self.u_net(params, t, T)
        return u1
    
    def r_net(self, params, t, T):
        u1, u2 = self.u_net(params, t, T)
        u1_t = grad(self.u1_net, argnums=1)(params, t, T)
        R1 = params['params']['R1']
        C1 = params['params']['C1']
        return u1_t + (u1 - V/u2)/(R1*C1)


    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):

        # Data loss
        u1_pred, u2_pred = self.u_pred_fn(params, self.t_ref, self.T_ref)
        data1_loss = jnp.mean((self.u1_ref - u1_pred) ** 2)
        data2_loss = jnp.mean((self.u2_ref - u2_pred) ** 2)

        # IC loss
        t0, T0 = get_initial_values()
        u1_t0, u2_t0 = self.u_pred_fn(params, t0, T0)
        R1 = params['params']['R1']
        ic = V/u2_t0 + V/R1
        ics_loss = jnp.mean((u1_t0 - ic) ** 2)

        # Res loss
        r_pred = self.r_pred_fn(params, batch[:, 0], batch[:, 1])
        res_loss = jnp.mean((r_pred) ** 2)

        #loss_dict = {"data1": data1_loss, "data2": data2_loss}
        #loss_dict = {"data1": data1_loss, "data2": data2_loss, "ics": ics_loss}
        #loss_dict = {"data1": data1_loss, "data2": data2_loss, "res": res_loss}
        loss_dict = {"data1": data1_loss, "data2": data2_loss, "ics": ics_loss, "res": res_loss}

        return loss_dict


    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u1_test, u2_test):
        u1_pred, u2_pred = vmap(self.u_net, (None, 0, 0))(params, self.t_ref, self.T_ref)
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
        u1_pred, u2_pred = vmap(self.model.u_net, (None, 0, 0))(params, self.model.t_ref, self.model.T_ref)

        # u1 plot in 3D
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.model.t_ref, self.model.T_ref, self.model.u1_ref, s=50, alpha=0.9, c='orange')
        points = jnp.column_stack((self.model.t_ref, self.model.T_ref, u1_pred))
        segments = jnp.split(points, jnp.unique(self.model.T_ref, return_index=True)[1][1:])
        line_segments = [list(zip(seg[:, 0], seg[:, 1], seg[:, 2])) for seg in segments]
        line_collection = Line3DCollection(line_segments, colors='black', linewidths=2)
        ax.add_collection(line_collection)
        self.log_dict["u1_pred"] = wandb.Image(fig)
        plt.close()

        # u2 plot in 2D
        fig = plt.figure(figsize=(6, 5))
        plt.scatter(self.model.T_ref, self.model.u2_ref, s=50, alpha=0.9, c='orange')
        plt.plot(self.model.T_ref, u2_pred, linewidth=8, c='black')
        self.log_dict["u2_pred"] = fig
        plt.close()
    
    def log_inv_params(self, params):
        self.log_dict["R1"] = params['params']['R1'][0]
        self.log_dict["C1"] = params['params']['C1'][0]
        

    def __call__(self, state, batch, u1_ref, u2_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.logging.log_errors:
            self.log_errors(state.params, u1_ref, u2_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params)
        
        if self.config.logging.log_inv_params:
            self.log_inv_params(state.params)

        return self.log_dict
    