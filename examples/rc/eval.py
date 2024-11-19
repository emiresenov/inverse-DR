import os

import ml_collections

import jax.numpy as jnp

import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint

import models
from utils import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    u_ref, t_star, x_star = get_dataset()
    u0 = u_ref[0]

    # Restore model
    model = models.Burgers(config, u0, t_star, x_star)
    ckpt_path = os.path.join(workdir, "ckpt", config.wandb.name)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    # Compute L2 error
    l2_error = model.compute_l2_error(params, u_ref)
    print("L2 error: {:.3e}".format(l2_error))

    u_pred = model.u_pred_fn(params, model.t_star, model.x_star)
    TT, XX = jnp.meshgrid(t_star, x_star, indexing="ij")

    plt.plot(TT, u_pred)
    plt.show()
