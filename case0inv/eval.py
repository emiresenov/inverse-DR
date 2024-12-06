import os

import ml_collections

import jax.numpy as jnp

import matplotlib.pyplot as plt

from jaxpi.utils import restore_checkpoint

import models
from utils import get_dataset


def evaluate(config: ml_collections.ConfigDict, workdir: str):
    u_ref, t_star = get_dataset()
    

    # Restore model
    model = models.CaseZero(config, u_ref, t_star)
    ckpt_path = os.path.join(workdir, config.wandb.name, "ckpt")
    ckpt_path = os.path.abspath(ckpt_path)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    u_pred = model.u_pred_fn(params, model.t_star)

    fig = plt.figure(figsize=(6, 5))
    plt.plot(model.t_star, u_pred)
    plt.scatter(model.t_star, model.u_ref, s=50, c='purple')

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "case0.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)

    