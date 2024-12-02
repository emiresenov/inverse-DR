import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "Case 1 inverse"
    wandb.name = "lr=1e-1"
    wandb.tag = None

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = 1
    arch.hidden_dim = 15
    arch.out_dim = 1
    arch.activation = "tanh"
    arch.periodicity = None
    arch.fourier_emb = None
    arch.reparam = None

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-1
    optim.decay_rate = 0.9
    optim.decay_steps = 1000
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 80000
    training.batch_size_per_device = 4096

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = None
    weighting.init_weights = ml_collections.ConfigDict({"data": 1.0, "ics": 1.0, "res": 1.0})
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    weighting.use_causal = False
    weighting.causal_tol = 1.0
    weighting.num_chunks = 32

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 10000
    saving.num_keep_ckpts = 50

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = False
    logging.log_grads = True
    logging.log_ntk = False
    logging.log_preds = True
    logging.log_inv_params = True
    
    # Inverse parameters
    config.inverse = inverse = ml_collections.ConfigDict()
    inverse.params = {
        'R0' : jnp.array([1.]), # true = 25
        'R1' : jnp.array([1.]), # true = 0.1
        'C1' : jnp.array([1.])  # true = 8 
    }

    # Input shape for initializing Flax models
    config.input_dim = 1

    # Integer for PRNG random seed.
    config.seed = 42

    return config
