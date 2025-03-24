import os
import ml_collections
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxpi.utils import restore_checkpoint
import models
from utils import get_dataset
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from utils import R1, C1

def evaluate(config: ml_collections.ConfigDict, workdir: str):
    t_star, T_star, u1_ref, u2_ref = get_dataset()
    
    '''
    Plot and save predictions
    '''

    # Restore model
    model = models.CaseOneField(config, t_star, T_star, u1_ref, u2_ref)
    ckpt_path = os.path.join(workdir, config.wandb.name, "ckpt")
    ckpt_path = os.path.abspath(ckpt_path)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    u1_pred, u2_pred = model.u_pred_fn(params, t_star, T_star)

    # u1 plot in 3D
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(t_star, T_star, u1_ref, s=50, alpha=0.9, c='orange')
    points = jnp.column_stack((t_star, T_star, u1_pred))
    segments = jnp.split(points, jnp.unique(T_star, return_index=True)[1][1:])
    line_segments = [list(zip(seg[:, 0], seg[:, 1], seg[:, 2])) for seg in segments]
    line_collection = Line3DCollection(line_segments, colors='black', linewidths=2)
    ax.add_collection(line_collection)
    ax.set_xlabel("Time (t)")
    ax.set_ylabel("Temperature (T)")
    ax.set_zlabel("Current (I)")
    plt.close()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "r01t_case1.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)



    '''
    Plot and save inverse parameter convergence
    '''

    '''# Initialize API
    api = wandb.Api()
    runs = api.runs(f"{config.wandb.project}")

    # Get the last run
    if runs:
        last_run = runs[-1] 
        run_id = last_run.id
        print(f"Last run ID: {run_id}")
    else:
        print("No runs found in the project.")
    
    history = last_run.history()

    # Convert to DataFrame for easy manipulation
    df = pd.DataFrame(history)'''
    '''history.to_csv("wandb_run_history.csv", index=False)
    print(df.head())  # Display the data
    print(df.columns.tolist())
    print(df['R'])'''


    '''plt.rc('font', family='serif')
    plt.rc('font', size=14)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)
    plt.rc('figure', titlesize=16)

    fig = plt.figure(figsize=(15, 8))


    linecol = 'deepskyblue'
    dashcolor = '#BB00BB'


    # Update x-axis ticks for all axes
    x_ticks = [0, 20, 40, 60, 80, 100]
    x_labels = ['0', '10k', '20k', '30k', '40k', '50k']



    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "r01tcase1params.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)'''