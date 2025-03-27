import os
import ml_collections
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib import rcParams
import matplotlib.lines as mlines
from jaxpi.utils import restore_checkpoint
import models
from utils import get_dataset, t_scale, C1, R1_const
import wandb
import pandas as pd


from matplotlib.ticker import FuncFormatter




def evaluate(config: ml_collections.ConfigDict, workdir: str):
    """
    Load dataset, restore the PINN model, predict current, and plot 
    both measurements and predictions in 3D with a publication-quality style.
    """
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)
    rcParams["font.family"] = "serif"
    rcParams["font.sans-serif"] = ["Arial"]  # or another preferred font

    t_star, T_star, u1_ref, u2_ref = get_dataset()

    model = models.CaseOneField(config, t_star, T_star, u1_ref, u2_ref)
    ckpt_path = os.path.join(workdir, config.wandb.name, "ckpt")
    ckpt_path = os.path.abspath(ckpt_path)
    model.state = restore_checkpoint(model.state, ckpt_path)
    params = model.state.params


    u1_pred, u2_pred = model.u_pred_fn(params, t_star, T_star)

    ### ------------------------
    # u1 PLOT
    ### ------------------------

    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')


    scatter_plot = ax.scatter(
        t_star, T_star, u1_ref, 
        s=50, 
        alpha=0.8, 
        c='orange', 
        edgecolor='k', 
        linewidth=0.5, 
        label="Measurements"
    )

    points = jnp.column_stack((t_star, T_star, u1_pred))
    unique_temps, temp_indices = jnp.unique(T_star, return_index=True)
    segments = jnp.split(points, temp_indices[1:])
    line_segments = [list(zip(seg[:, 0], seg[:, 1], seg[:, 2])) for seg in segments]

    line_collection = Line3DCollection(
        line_segments, 
        colors='blue', 
        linewidths=2
    )

    
    # Format axis labels only
    from matplotlib.ticker import FuncFormatter
    def scale_temp(x, pos):
        return f"{x * t_scale:.0f}"
    ax.yaxis.set_major_formatter(FuncFormatter(scale_temp))
    ax.add_collection(line_collection)

    line_proxy = mlines.Line2D([], [], color='blue', linewidth=2, label='PINN Predictions')

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (K)")
    ax.set_zlabel("Current (A)")

    ax.view_init(elev=20, azim=-60)  # Adjust for best visibility

    ax.legend([scatter_plot, line_proxy], ["Measurements", "PINN Predictions"], loc="upper left")

    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    fig_path = os.path.join(save_dir, "r01t_case1.pdf")
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    ### ------------------------
    # COMBINED R0(T) & R1(T) PLOT
    ### ------------------------

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

    history = last_run.history()
    df = pd.DataFrame(history)


    fig = plt.figure(figsize=(6, 4), dpi=300)
    plt.rc('font', family='serif')
    line_color_r0 = "#1f77b4"  # Professional blue
    scatter_color_r0 = "#ff7f0e"  # Rich orange
    line_color_r1 = "#2ca02c"  # Green
    scatter_color_r1 = "#d62728"  # Red

    # R0(T)
    plt.plot(T_star, u2_pred, label='Learned $\hat{R}_0(T)$', color=line_color_r0, zorder=1)
    plt.scatter(T_star, model.u2_ref, label='Sampled $R_0(T)$',
                color=scatter_color_r0, edgecolor="black", s=80, marker='o', zorder=2)

    # R1(T) - scaled
    final_r1 = df["R1"].dropna().values[-1]
    scaled_u2_ref = model.u2_ref * R1_const
    scaled_u2_pred = u2_pred * final_r1

    plt.plot(T_star, scaled_u2_pred, label='Learned $\hat{R}_1(T)$', color=line_color_r1, zorder=3)
    plt.scatter(T_star, scaled_u2_ref, label='Sampled $R_1(T)$',
                color=scatter_color_r1, edgecolor="black", s=80, marker='s', zorder=4)

    # Axes and formatting
    plt.gca().xaxis.set_major_formatter(FuncFormatter(scale_temp))
    plt.xlabel('Temperature (K)', fontsize=16, labelpad=10)
    plt.ylabel('Resistance (Ω)', fontsize=16, labelpad=10)

    plt.grid(visible=True, linestyle='--', linewidth=0.6, alpha=0.5)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.legend(fontsize=13, frameon=True, loc='upper right', framealpha=0.8, edgecolor='gray')

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "r01(T).pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)












    '''    ### ------------------------
    # C1 PLOT
    ### ------------------------
    # Plot style settings
    plt.rc('font', family='serif')
    plt.rc('font', size=14)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('legend', fontsize=12)
    plt.rc('figure', titlesize=16)

    # Define colors
    linecol = 'deepskyblue'
    dashcolor = '#BB00BB'

    # Plot
    fig, ax = plt.subplots(figsize=(3, 2.5))
    sns.lineplot(data=df, x=df.index, y='C1', label='Estimated $C1$', ax=ax, linewidth=3, color=linecol)
    ax.axhline(C1, color=dashcolor, linestyle='--', label='True $C1$', linewidth=3, dashes=(5, 6))
    ax.set_xlabel('Training step')
    ax.set_ylabel('Capacitance (farad)')
    ax.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.5)

    # Legend
    true_c1 = mlines.Line2D([], [], color=dashcolor, linestyle=(0, (3.5, 2)), linewidth=2.5, label='True $C1$')
    estimated_c1 = mlines.Line2D([], [], color=linecol, linewidth=2.5, label='Estimated $C1$')
    ax.legend(
        handles=[estimated_c1, true_c1],
        loc='upper right',           # or 'upper left', 'lower left', etc.
        frameon=True,                # box around the legend
        framealpha=1.0,              # solid box background
        edgecolor='gray',           # box border color
        fontsize='small'            # optional: to keep legend compact
    )

    # Custom x-ticks (NEEDS TO BE HARDCODED)
    x_ticks = [0, 10, 20, 30, 40, 50]
    x_labels = ['0', '10k', '20k', '30k', '40k', '50k']
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, "c1_plot.pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)'''