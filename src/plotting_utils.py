import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter

font_size = 14
label_font_size = 18

font_style = "Arial"
mpl.rcParams['font.family'] = font_style
mpl.rcParams['font.size'] = font_size 

def marginal_plot(parameter_names, samples_flattened, parameter_x_axis_limits, parameter_x_axis_interval, parameter_y_axis_limits, parameter_y_axis_interval, color, outname):

    num_params = len(parameter_names)
    
    fig, axes = plt.subplots(num_params, 1, figsize=(6, 3 * num_params))

    for i, param in enumerate(parameter_names):
        ax = axes[i] if num_params > 1 else axes
        ax.hist(samples_flattened[:, i], bins=50, density=True, alpha=0.4, color=color[i])

        # Set x-axis limits and intervals
        ax.set_xticks(np.arange(parameter_x_axis_limits[i][0], parameter_x_axis_limits[i][1]+parameter_x_axis_interval[i], parameter_x_axis_interval[i]))
        
        ax.set_xlim(parameter_x_axis_limits[i])
        ax.set_ylim(parameter_y_axis_limits[i])
        
        ax.grid(True)   
        ax.set_xlabel(param +' value', fontsize=label_font_size, fontname=font_style)
        ax.set_ylabel('Density', fontsize=label_font_size, fontname=font_style)

    plt.tight_layout()
    plt.savefig(outname, dpi=300)

def trace_plot(parameter_names, samples, x_axis_limits, parameter_y_axis_limits, parameter_y_axis_interval, alpha, outname):
    num_params = len(parameter_names)
    num_chains = samples.shape[1]

    # Create subplots for each parameter
    fig, axes = plt.subplots(num_params, 1, figsize=(6, 3 * num_params), squeeze=False)

    for i, param in enumerate(parameter_names):
        ax = axes[i, 0]  # Single row of subplots
        for chain in range(num_chains):
            ax.plot(samples[:, chain, i], alpha=alpha, label=f'Chain {chain + 1}')
            ax.set_yticks(np.arange(parameter_y_axis_limits[i][0], parameter_y_axis_limits[i][1]+parameter_y_axis_interval[i], parameter_y_axis_interval[i]))

        ax.grid(True)
        ax.set_xlim(x_axis_limits)
        ax.set_ylim(parameter_y_axis_limits[i])
        ax.set_ylabel(param+' value', fontsize=label_font_size, fontname=font_style)
        ax.set_xlabel('Iteration', fontsize=label_font_size, fontname=font_style)

    plt.tight_layout()
    plt.savefig(outname, dpi=300)