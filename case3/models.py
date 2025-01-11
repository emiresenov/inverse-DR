from functools import partial

import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, tree_leaves, jacobian

from jaxpi.models import InverseIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from utils import V

from matplotlib import pyplot as plt


class CaseThree(InverseIVP):
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
        return u[0], u[1], u[2]

    # Calculate gradients
    def grad_net(self, params, t):
        grads = jacobian(self.u_net, argnums=1)(params, t)
        u1_t = grads[0]
        u2_t = grads[1]
        u3_t = grads[2]
        return u1_t, u2_t, u3_t

    # Diff eq prediction (residual)
    def r_net(self, params, t):
        u1, u2, u3 = self.u_net(params, t)
        u1_t, u2_t, u3_t = self.grad_net(params, t)
        R0 = params['params']['R0']
        R1 = params['params']['R1']
        C1 = params['params']['C1']
        R2 = params['params']['R2']
        C2 = params['params']['C2']
        R3 = params['params']['R3']
        C3 = params['params']['C3']
        return u1_t + (u1-(V/R0))/(R1*C1), u2_t + u2/(R2*C2), u3_t + u3/(R3*C3)

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Initial condition loss
        R0 = params['params']['R0']
        R1 = params['params']['R1']
        R2 = params['params']['R2']
        R3 = params['params']['R3']
        ic1 = V/R0 + V/R1
        ic2 = V/R2
        ic3 = V/R3
        u0_pred_1, u0_pred_2, u0_pred_3 = self.u_net(params, self.t0)
        ic1_loss = jnp.mean((u0_pred_1 - ic1) ** 2)
        ic2_loss = jnp.mean((u0_pred_2 - ic2) ** 2)
        ic3_loss = jnp.mean((u0_pred_3 - ic3) ** 2)

        # Residual loss
        r1_pred, r2_pred, r3_pred = vmap(self.r_net, (None, 0))(params, batch[:, 0])
        res1_loss = jnp.mean((r1_pred) ** 2)
        res2_loss = jnp.mean((r2_pred) ** 2)
        res3_loss = jnp.mean((r3_pred) ** 2)

        # Data loss
        u1_pred, u2_pred, u3_pred = self.u_pred_fn(params, self.t_star)
        u_pred = u1_pred + u2_pred + u3_pred
        data_loss = jnp.mean((self.u_ref - u_pred) ** 2)

        #l1_penalty = 1e-1 * sum(jnp.sum(jnp.abs(w)) for w in tree_leaves(params))
        #loss_dict = {"data": data_loss + l1_penalty, "ics": ics_loss, "res": res_loss}

        loss_dict = {
            "data": data_loss, 
            "ic1": ic1_loss, 
            "ic2": ic2_loss, 
            "ic3": ic3_loss, 
            "res1": res1_loss, 
            "res2": res2_loss,
            "res3": res3_loss
            }
        return loss_dict


    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, u_test):
        u1_pred, u2_pred, u3_pred = self.u_pred_fn(params, self.t_star)
        u_pred = u1_pred + u2_pred + u3_pred
        error = jnp.linalg.norm(u_pred - u_test) / jnp.linalg.norm(u_test)
        return error


class CaseThreeEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, u_ref):
        l2_error = self.model.compute_l2_error(params, u_ref)
        self.log_dict["l2_error"] = l2_error

    def log_preds(self, params):
        u_pred_1, u_pred_2, u_pred_3 = self.model.u_pred_fn(params, self.model.t_star)
        fig = plt.figure(figsize=(6, 5))
        plt.scatter(self.model.t_star, self.model.u_ref, s=50, alpha=0.9, c='orange')
        plt.plot(self.model.t_star, u_pred_1 + u_pred_2 + u_pred_3, linewidth=8, c='black')
        self.log_dict["u_pred"] = fig
        plt.close()
    
    def log_inv_params(self, params):
        self.log_dict["R0"] = params['params']['R0'][0]
        self.log_dict["R1"] = params['params']['R1'][0]
        self.log_dict["C1"] = params['params']['C1'][0]
        self.log_dict["R2"] = params['params']['R2'][0]
        self.log_dict["C2"] = params['params']['C2'][0]
        self.log_dict["R3"] = params['params']['R3'][0]
        self.log_dict["C3"] = params['params']['C3'][0]
        

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
