from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, tree_leaves, random

from jaxpi.models import InverseIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree
from jax.tree_util import tree_flatten, tree_unflatten

from matplotlib import pyplot as plt

from utils import V, get_initial_values

from subnets import R0Net


class CaseOneField(InverseIVP):
    def __init__(self, config, u_ref, R0_ref, t_star, T_star):

        self.R0_net = R0Net()
        R0_params = self.R0_net.init(random.PRNGKey(1234), jnp.array([1.]))
        leaves, structure = tree_flatten(R0_params)
        config.inverse.params['R0_params'] = leaves
        self.R0_struct = structure # Save struct for apply calls with updated leaves

        super().__init__(config)

        self.u_ref = u_ref
        self.R0_ref = R0_ref
        self.t_star = t_star
        self.T_star = T_star

        self.t0, self.T0 = get_initial_values() 

        self.u_pred_fn = vmap(self.u_net, (None, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0))
        self.R0_pred_fn = vmap(self.R0_pred, (None, 0))


    def u_net(self, params, t):
        z = jnp.stack([t])
        u = self.state.apply_fn(params, z)
        return u[0]

    def grad_net(self, params, t):
        u_t = grad(self.u_net, argnums=1)(params, t)
        return u_t

    def r_net(self, params, t, T):
        u = self.u_net(params, t)
        u_t = grad(self.u_net, argnums=1)(params, t)
        R0 = self.R0_pred(params, T)
        R1 = params['params']['R1']
        C1 = params['params']['C1']
        return u_t + (u - V/R0)/(R1*C1)
    
    def R0_pred(self, params, T):
        leaves = params['params']['R0_params']
        R0_params = tree_unflatten(self.R0_struct, leaves)
        R0 = self.R0_net.apply(R0_params, T)
        return R0

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Initial condition loss
        R1 = params['params']['R1']
        R0 = self.R0_pred_fn(params, self.T0)
        ic = V/R0 + V/R1
        u0_pred = self.u_pred_fn(params, self.t0)
        ics_loss = jnp.mean((u0_pred - ic) ** 2)

        # Residual loss
        r_pred = self.r_pred_fn(params, batch[:, 0], batch[:, 1]) # TODO: Confirm
        res_loss = jnp.mean((r_pred) ** 2)

        # Data loss
        u_pred = self.u_pred_fn(params, self.t_star)
        data_loss = jnp.mean((self.u_ref - u_pred) ** 2)

        # Subnet loss
        subnet_loss = jnp.mean((self.R0_ref - R0) ** 2)

        #l1_penalty = 1e-2 * sum(jnp.sum(jnp.abs(w)) for w in tree_leaves(params))
        #loss_dict = {"data": data_loss + l1_penalty, "ics": ics_loss, "res": res_loss}
        
        loss_dict = {"data": data_loss, "ics": ics_loss, "res": res_loss, "subnet": subnet_loss}

        return loss_dict


    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred = self.u_pred_fn(params, self.t_star)
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error


class CaseOneFieldEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params):
        u_pred = self.model.u_pred_fn(params, self.model.t_star)
        fig = plt.figure(figsize=(6, 5))
        plt.scatter(self.model.t_star, self.model.u_ref, s=50, alpha=0.9, c='orange')
        plt.plot(self.model.t_star, u_pred, linewidth=8, c='black')
        self.log_dict["u_pred"] = fig
        plt.close()
    
    def log_inv_params(self, params):
        self.log_dict["R1"] = params['params']['R1'][0]
        self.log_dict["C1"] = params['params']['C1'][0]
        
    def log_subnet(self, params):
        R0_pred = self.model.R0_pred_fn(params, self.model.T0)
        fig = plt.figure(figsize=(6, 5))
        plt.scatter(self.model.T0, self.model.R0_ref, s=50, alpha=0.9, c='orange')
        plt.plot(self.model.T0, R0_pred, linewidth=8, c='black')
        self.log_dict["R0_pred"] = fig
        plt.close()

    def __call__(self, state, batch, u_ref):
        self.log_dict = super().__call__(state, batch)

        if self.config.weighting.use_causal:
            _, causal_weight = self.model.res_and_w(state.params, batch)
            self.log_dict["cas_weight"] = causal_weight.min()

        if self.config.logging.log_errors:
            self.log_errors(state.params, u_ref)

        if self.config.logging.log_preds:
            self.log_preds(state.params)
        
        if self.config.logging.log_inv_params:
            self.log_inv_params(state.params)

        if self.config.logging.log_subnet:
            self.log_subnet(state.params)

        return self.log_dict
    