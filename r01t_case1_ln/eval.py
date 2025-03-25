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
from utils import get_dataset, t_scale


from matplotlib.ticker import FuncFormatter

# Custom formatter to scale T axis tick labels by 330.15
def scale_temp(x, pos):
    return f"{x * t_scale:.1f}"



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
        return f"{x * t_scale:.1f}"
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


    # u2 plot in 2D
    fig = plt.figure(figsize=(6, 4), dpi=300)
    plt.rc('font', family='serif')
    line_color = "#1f77b4"  # Professional blue
    scatter_color = "#ff7f0e"  # Rich orange    

    plt.plot(T_star, u2_pred, label='Learned $\hat{R}_0(T)$', color=line_color, zorder=1)
    plt.scatter(T_star, model.u2_ref, label='Sampled $R_0(T)$',
                color=scatter_color, edgecolor="black", s=80, marker='o', zorder=2)

    # Apply temperature scaling formatter to x-axis
    plt.gca().xaxis.set_major_formatter(FuncFormatter(scale_temp))

    plt.xlabel('Temperature (K)', fontsize=16, labelpad=10)
    plt.ylabel('Resistance (Ω)', fontsize=16, labelpad=10)

    plt.grid(visible=True, linestyle='--', linewidth=0.6, alpha=0.5)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    plt.legend(fontsize=14, frameon=True, loc='upper right', framealpha=0.8, edgecolor='gray')

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    fig_path = os.path.join(save_dir, "r0(T).pdf")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)