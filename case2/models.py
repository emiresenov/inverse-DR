from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, tree_leaves

from jaxpi.models import InverseIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt


class CaseTwo(InverseIVP):
    def __init__(self, config, u_ref, t_star):
        super().__init__(config)
        self.t_star = t_star
        self.u_ref = u_ref

        self.t0 = t_star[0]

        # Vectorizing functions over multiple data points
        self.u_pred_fn = vmap(self.u_net, (None, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0))

    # Prediction function for a given point in the domain
    def u_net(self, params, t):
        z = jnp.stack([t])
        u = self.state.apply_fn(params, z)
        return u[0], u[1]

    # Calculate gradients
    def grad_net(self, params, t):
        u1_t = grad(self.u_net, argnums=1)(params, t)
        u2_t = grad(self.u_net, argnums=2)(params, t)
        return u1_t, u2_t

    # Diff eq prediction (residual)
    def r_net(self, params, t):
        V = self.config.constants.V
        u1, u2 = self.u_net(params, t)
        u1_t, u2_t = grad(self.u_net, argnums=1)(params, t)
        R0 = params['params']['R0']
        R1 = params['params']['R1']
        C1 = params['params']['C1']
        R2 = params['params']['R2']
        C2 = params['params']['C2']
        return u1_t+(1/(R1*C1))*(u1-V/R0), u2_t+u2/(R2*C2)

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Initial condition loss
        V = self.config.constants.V
        R0 = params['params']['R0']
        R1 = params['params']['R1']
        R2 = params['params']['R2']
        ic = V/R0 + V/R1 + V/R2
        u0_pred_1, u0_pred_2 = self.u_net(params, self.t0) # Alternative: use self.u0
        u0_pred = u0_pred_1 + u0_pred_2
        ics_loss = jnp.mean((u0_pred - ic) ** 2)

        # Residual loss
        r1_pred, r2_pred = vmap(self.r_net, (None, 0))(params, batch[:, 0])
        res1_loss = jnp.mean((r1_pred) ** 2)
        res2_loss = jnp.mean((r2_pred) ** 2)

        # Data loss
        u1_pred, u2_pred = self.u_pred_fn(params, self.t_star)
        u_pred = u1_pred + u2_pred
        data_loss = jnp.mean((self.u_ref - u_pred) ** 2)

        #l1_penalty = 1e-1 * sum(jnp.sum(jnp.abs(w)) for w in tree_leaves(params))
        #loss_dict = {"data": data_loss + l1_penalty, "ics": ics_loss, "res": res_loss}

        loss_dict = {"data": data_loss, "ics": ics_loss, "res1": res1_loss, "res2": res2_loss}
        return loss_dict


    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u1_pred, u2_pred = self.u_pred_fn(params, self.t_star)
        u_pred = u1_pred + u2_pred
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error


class CaseTwoEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params):
        u1_pred, u2_pred = self.model.u_pred_fn(params, self.model.t_star)
        u_pred = u1_pred + u2_pred
        fig = plt.figure(figsize=(6, 5))
        plt.scatter(self.model.t_star, self.model.u_ref, s=50, alpha=0.65, c='orange')
        plt.plot(self.model.t_star, u_pred, linewidth=4, c='black')
        self.log_dict["u_pred"] = fig
        plt.close()
    
    def log_inv_params(self, params):
        self.log_dict["R0"] = params['params']['R0'][0]
        self.log_dict["R1"] = params['params']['R1'][0]
        self.log_dict["C1"] = params['params']['C1'][0]
        self.log_dict["R2"] = params['params']['R2'][0]
        self.log_dict["C2"] = params['params']['C2'][0]
        

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

        return self.log_dict
