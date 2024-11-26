import os
import time
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map

import ml_collections
from absl import logging
import wandb

from jaxpi.samplers import UniformSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint

import models
from utils import get_dataset


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
    # Initialize W&B
    #wandb_config = config.wandb
    #wandb.init(project=wandb_config.project, name=wandb_config.name)

    # Initialize logger
    logger = Logger()

    # Get dataset
    u_ref, t_star = get_dataset()

    t0 = t_star[0]
    t1 = t_star[-1]

    # Define domain
    dom = jnp.array([[t0, t1]])

    # Define residual sampler
    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

    # Initialize model
    model = models.CaseZero(config, t_star)

    # Initialize evaluator
    evaluator = models.CaseZeroEvaluator(config, model)

    print("Waiting for JIT...")
    start_time = time.time()
    for step in range(config.training.max_steps):
        batch = next(res_sampler)
        model.state = model.step(model.state, batch)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, u_ref)

                end_time = time.time()
                logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time

        '''# Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                ckpt_path = os.path.join(os.getcwd(), config.wandb.name, "ckpt")
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)'''

    u_pred = model.u_pred_fn(state.params, model.t_star)
    # plot
    fig = plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(t_star, u_ref)
    plt.title("Exact")
    plt.tight_layout()

    plt.subplot(1, 3, 2)
    plt.plot(t_star, u_pred)
    plt.title("Predicted")
    plt.tight_layout()

    plt.subplot(1, 3, 3)
    plt.plot(t_star, jnp.abs(u_ref - u_pred))
    plt.title("Absolute error")
    plt.tight_layout()
    
    print(f"predicted tau: {state.params['params']['tau']}")

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "case0inv.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    
    return model
