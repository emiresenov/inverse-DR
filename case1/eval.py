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

from utils import R0, R1, C1

def evaluate(config: ml_collections.ConfigDict, workdir: str):
    u_ref, t_star = get_dataset()
    
    '''
    Plot and save predictions
    '''

    # Restore model
    model = models.CaseOne(config, u_ref, t_star)
    ckpt_path = os.path.join(workdir, config.wandb.name, "ckpt")
    ckpt_path = os.path.abspath(ckpt_path)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params

    u_pred = model.u_pred_fn(params, model.t_star)

    fig = plt.figure(figsize=(6, 4), dpi=300)
    plt.rc('font', family='serif')
    line_color = "#1f77b4"  # Professional blue
    scatter_color = "#ff7f0e"  # Rich orange    
    plt.plot(model.t_star, u_pred, label='Prediction', color=line_color, linewidth=2.5, linestyle='-', zorder=1)
    plt.scatter(model.t_star, model.u_ref, label='Measurement', color=scatter_color, edgecolor="black", s=80, marker='o', zorder=2)
    
    plt.xlabel('Time (s)', fontsize=16, labelpad=10)
    plt.ylabel('Current (A)', fontsize=16, labelpad=10)
    plt.title('PINN Solution', fontsize=18, weight='bold', pad=15)

    plt.grid(visible=True, linestyle='--', linewidth=0.6, alpha=0.5)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    # Legend
    plt.legend(fontsize=14, frameon=True, loc='upper right', framealpha=0.8, edgecolor='gray')

    # Remove unnecessary spines
    for spine in ['top', 'right']:
        plt.gca().spines[spine].set_visible(False)

    # Tight layout
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "case1.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)



    '''
    Plot and save inverse parameter convergence
    '''

    # Initialize API
    api = wandb.Api()
    runs = api.runs(f"{config.wandb.project}")

    # Get the last run
    if runs:
        last_run = runs[-1] 
        run_id = last_run.id
        print(f"Last run ID: {run_id}")
    else:
        print("No runs found in the project.")
    
    history = last_run.history()  # Replace with your keys

    # Convert to DataFrame for easy manipulation
    df = pd.DataFrame(history)
    '''history.to_csv("wandb_run_history.csv", index=False)
    print(df.head())  # Display the data
    print(df.columns.tolist())
    print(df['R'])'''


    fig = plt.figure(figsize=(11, 4), dpi=300)

    plt.rc('font', family='serif')
    plt.rc('font', size=14)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)
    plt.rc('figure', titlesize=16)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharex=True)

    linecol = 'deepskyblue'
    dashcolor = '#BB00BB'

    sns.lineplot(data=df, x=df.index, y='R0', label='Estimated $R0$', ax=axes[0], linewidth=3, color=linecol)
    axes[0].axhline(R0, color=dashcolor, linestyle='--', label='True $R0$', linewidth=3, dashes=(5, 6))  # Adjust dashes here
    axes[0].set_xlabel('Training step')
    axes[0].set_ylabel('Resistance $R$ ($\Omega$)')
    axes[0].grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)

    sns.lineplot(data=df, x=df.index, y='C1', label='Estimated $C1$', ax=axes[1], linewidth=3, color=linecol)
    axes[1].axhline(C1, color=dashcolor, linestyle='--', label='True $C1$', linewidth=3, dashes=(5, 6))  # Adjust dashes here
    axes[1].set_xlabel('Training step')
    axes[1].set_ylabel('Capacitance $C$ (farad)')
    axes[1].grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)

    sns.lineplot(data=df, x=df.index, y='R1', label='Estimated $R1$', ax=axes[2], linewidth=3, color=linecol)
    axes[2].axhline(R1, color=dashcolor, linestyle='--', label='True $R1$', linewidth=3, dashes=(5, 6))  # Adjust dashes here
    axes[2].set_xlabel('Training step')
    axes[2].set_ylabel('Resistance $R$ ($\Omega$)')
    axes[2].grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)


    # Custom Legend
    true_r0 = mlines.Line2D([], [], color=dashcolor, linestyle=(0, (3.5, 2)), linewidth=2.5, label='True $R0$')  # Custom dashed line for legend
    true_c1 = mlines.Line2D([], [], color=dashcolor, linestyle=(0, (3.5, 2)), linewidth=2.5, label='True $C1$')  # Custom dashed line for legend
    true_r1 = mlines.Line2D([], [], color=dashcolor, linestyle=(0, (3.5, 2)), linewidth=2.5, label='True $R1$')  # Custom dashed line for legend
    estimated_r0 = mlines.Line2D([], [], color=linecol, linewidth=2.5, label='Estimated $R0$')
    estimated_c1 = mlines.Line2D([], [], color=linecol, linewidth=2.5, label='Estimated $C1$')
    estimated_r1 = mlines.Line2D([], [], color=linecol, linewidth=2.5, label='Estimated $R1$')

    axes[0].legend(handles=[estimated_r0, true_r0], loc='upper center', bbox_to_anchor=(0.5, 1.25), frameon=False)
    axes[1].legend(handles=[estimated_c1, true_c1], loc='upper center', bbox_to_anchor=(0.5, 1.25), frameon=False)
    axes[2].legend(handles=[estimated_r1, true_r1], loc='upper center', bbox_to_anchor=(0.5, 1.25), frameon=False)


    '''Doing this manually because dynamic handling is messy'''
    # Update x-axis ticks for both plots
    x_ticks = [0, 10, 20, 30]  # Original tick positions
    x_labels = ['0', '10k', '20k', '30k']  # New labels

    axes[0].set_xticks(x_ticks)
    axes[0].set_xticklabels(x_labels)
    axes[1].set_xticks(x_ticks)
    axes[1].set_xticklabels(x_labels)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
    
    # Tight layout
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "case1params.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)