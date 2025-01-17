from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, tree_leaves, random

from jaxpi.models import InverseIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt

from utils import V

from subnets import ResistanceNet


class CaseZeroField(InverseIVP):
    def __init__(self, config, u_ref, t_star):

        self.subnet_R = ResistanceNet()
        subnet_R_params = self.subnet_R.init(random.PRNGKey(1234), t_star)
        config.inverse.params['R_params'] = subnet_R_params

        super().__init__(config)

        self.t_star = t_star
        self.u_ref = u_ref

        self.t0 = t_star[0]

        self.subnet_R_fn = vmap(self.subnet_R.apply, (None, 0))
        self.u_pred_fn = vmap(self.u_net, (None, 0))
        self.r_pred_fn = vmap(self.r_net, (None, 0))

    def u_net(self, params, t):
        z = jnp.stack([t])
        u = self.state.apply_fn(params, z)
        return u[0]

    def grad_net(self, params, t):
        u_t = grad(self.u_net, argnums=1)(params, t)
        return u_t

    # Diff eq prediction
    def r_net(self, params, t):
        u = self.u_net(params, t)
        u_t = grad(self.u_net, argnums=1)(params, t)
        R_params = params['params']['R_params']
        R = self.subnet_R.apply(R_params, t)
        C = params['params']['C']
        return u_t + u/(R*C)

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Initial condition loss
        R_params = params['params']['R_params']
        R = self.subnet_R.apply(R_params, self.t0)
        ic = V/R
        u0_pred = self.u_net(params, self.t0)
        ics_loss = jnp.mean((u0_pred - ic) ** 2)

        # Residual loss
        if self.config.weighting.use_causal == True:
            l, w = self.res_and_w(params, batch)
            res_loss = jnp.mean(l * w)
        else:
            r_pred = vmap(self.r_net, (None, 0))(params, batch[:, 0])
            res_loss = jnp.mean((r_pred) ** 2)

        # Data loss
        u_pred = self.u_pred_fn(params, self.t_star)
        data_loss = jnp.mean((self.u_ref - u_pred) ** 2)
        
        #l1_penalty = 1e-1 * sum(jnp.sum(jnp.abs(w)) for w in tree_leaves(params))
        #loss_dict = {"data": data_loss + l1_penalty, "ics": ics_loss, "res": res_loss}
        
        loss_dict = {"data": data_loss, "ics": ics_loss, "res": res_loss}
        
        return loss_dict


    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u_pred = self.u_pred_fn(params, self.t_star)
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error


class CaseZeroEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params):
        u_pred = self.model.u_pred_fn(params, self.model.t_star)
        fig = plt.figure(figsize=(6, 5))
        plt.scatter(self.model.t_star, self.model.u_ref, s=50, c='purple')
        plt.plot(self.model.t_star, u_pred, linewidth=3)
        self.log_dict["u_pred"] = fig
        plt.close()
    
    def log_inv_params(self, params):
        R = params['params']['R'][0]
        C = params['params']['C'][0]
        self.log_dict["R"] = R
        self.log_dict["C"] = C
        

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
