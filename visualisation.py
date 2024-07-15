import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import seaborn as sns
import pandas as pd
from functions import extract_data
from initialise import init_states

def visualise(model, output):
    """
    Configure and visualize the model using the specified output.

    Parameters:
    - model: The model object to be visualized.
    - output: The output data from the model.

    This function sets up the visualization configuration and then generates an animation.
    """
    visualization_config = {
        'layout': 'fr',
        'plot_interval': 5,
        'plot_variable': 'SWB',
        'variable_limits': {
            'SWB': [0, 10],
        },
        'color_scale': 'RdBu',
        'show_plot': True,
        'plot_output': 'animations/SWB.gif',
        'plot_title': 'SWB',
    }

    model.configure_visualization(visualization_config, output)
    model.visualize('animation')

def plot_stoch_param(data, param_name, param_steps, intervention_timesteps, int_var, title_add=""):
    """
    Plot stochastic parameter changes over time.

    Parameters:
    - data: The data to plot.
    - param_name: The name of the parameter being varied.
    - param_steps: The steps of the parameter.
    - intervention_timesteps: Timesteps where interventions occur.
    - int_var: Variables indicating the type of intervention.
    - title_add: Additional title text for the plot.

    This function generates a line plot with shaded standard deviation areas.
    """
    line_color = "black"
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:gray']
    added_to_legend = []
    for index, param_step in enumerate(data):
        stds = np.std(param_step, axis=0)
        means = np.mean(param_step, axis=0)
        plt.fill_between(np.linspace(0, len(means), len(means)), means-stds, means+stds, color=colors[index], alpha=0.1)
        plt.plot(means, label=f"{param_name}: {param_steps[index]:.2f}", color=colors[index])
    for index, intervention in enumerate(intervention_timesteps):
        if int_var[index] == "fin":
            line_color = "green"
        elif int_var[index] == "nonfin":
            line_color = "blue"
        if int_var[index] not in added_to_legend:
            plt.axvline(x=intervention, color=line_color, linestyle='--', label=int_var[index] + " shock")
            added_to_legend.append(int_var[index])
        else:
            plt.axvline(x=intervention, color=line_color, linestyle='--')
    plt.legend()
    plt.title("SWB over time", fontsize=16)
    plt.xlabel("Iteration", fontsize=14)
    plt.ylabel("Average SWB", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.savefig(f"figures/OFAT/{title_add}{param_name}/SWB.pdf", dpi=300)
    plt.clf()
    return

def plot_stoch_components(fin, exp_fin, nonfin, exp_nonfin, RFC, exp_RFC, soccap, exp_soccap, param_name, param_steps, intervention_timesteps, int_var, title_add=""):
    """
    Plot stochastic components of the model over time.

    Parameters:
    - fin, exp_fin: Financial data and expected financial data.
    - nonfin, exp_nonfin: Non-financial data and expected non-financial data.
    - RFC, exp_RFC: RFC data and expected RFC data.
    - soccap, exp_soccap: Social capital data and expected social capital data.
    - param_name: The name of the parameter being varied.
    - param_steps: The steps of the parameter.
    - intervention_timesteps: Timesteps where interventions occur.
    - int_var: Variables indicating the type of intervention.
    - title_add: Additional title text for the plot.

    This function generates line plots with shaded standard deviation areas for multiple components.
    """
    line_color = "black"
    title_names = ["fin", "nonfin", "RFC", "soccap"]
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:brown', 'tab:pink', 'tab:gray']
    for index, data_tuple in enumerate([(fin, exp_fin), (nonfin, exp_nonfin), (RFC, exp_RFC), (soccap, exp_soccap)]):
        plt.figure(figsize=(12,8))
        for param_index, param_step in enumerate(data_tuple[0]):
            stds = np.std(param_step, axis=0)
            means = np.mean(param_step, axis=0)
            exp_means = np.mean(data_tuple[1][param_index], axis=0)
            plt.fill_between(np.linspace(0, len(means), len(means)), means-stds, means+stds, alpha=0.1, color=colors[param_index])
            plt.plot(means, linestyle='-', marker='.', label=f"{param_name}: {param_steps[param_index]:.2f}", color=colors[param_index])
            plt.plot(exp_means, linestyle='dashed', color=colors[param_index])
        for intervention_index, intervention in enumerate(intervention_timesteps):
            if int_var[intervention_index] == "fin":
                line_color = "green"
            elif int_var[intervention_index] == "nonfin":
                line_color = "blue"
            plt.axvline(x=intervention, color=line_color, linestyle='--')
        plt.title(f"{title_names[index]} : {param_name}")
        plt.xlabel("Iteration")
        plt.ylabel(f"Average {title_names[index]}")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f"figures/OFAT/{title_add}{param_name}/{title_names[index]}", bbox_inches='tight')
        plt.clf()
        plt.close()
    return

def plot_avg(means, stds, title_add=""):
    """
    Plot average values with standard deviation over time.

    Parameters:
    - means: The mean values to plot.
    - stds: The standard deviation values to plot.
    - title_add: Additional title text for the plot.

    This function generates line plots with shaded standard deviation areas for multiple variables.
    """
    labels = ["SWB", "Fin", "Nonfin", "RFC", "Soccap"]
    line_widths = [2, 1, 1, 1, 1]
    colours = ["c", "g", "b", "r", "purple"]
    for index, state_means in enumerate(means):
        state_stds = stds[index] / np.mean(state_means)
        state_means = state_means / np.mean(state_means)
        plt.plot(state_means, label=labels[index], color = colours[index], linewidth=line_widths[index])
        plt.fill_between(np.linspace(0, len(state_means), len(state_means)), state_means-state_stds, state_means+state_stds, color = colours[index], alpha=0.1)
    plt.title('Drivers and SWB over time', fontsize=16)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend()

    # Rotate x-axis labels to be vertical and set font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save and close figure
    plt.savefig(f"figures/avg_{title_add}.pdf", dpi=300)
    plt.clf()
    plt.close()

def plot_agent(mean_data, std_data, title_add=""):
    """
    Plot agent data with standard deviation over time.

    Parameters:
    - mean_data: The mean values to plot.
    - std_data: The standard deviation values to plot.
    - title_add: Additional title text for the plot.

    This function generates line plots with shaded standard deviation areas for multiple variables.
    """
    labels = ["SWB", "Fin", "Nonfin", "RFC", "Soccap"]
    line_widths = [2, 1, 1, 1, 1]
    colours = ["c", "g", "b", "r", "purple"]
    for index, state_means in enumerate(mean_data):
        state_stds = std_data[index] / np.mean(state_means)
        state_means = state_means / np.mean(state_means)
        plt.plot(state_means, label=labels[index], color = colours[index], linewidth=line_widths[index])
        plt.fill_between(np.linspace(0, len(state_means), len(state_means)), state_means-state_stds, state_means+state_stds, color = colours[index], alpha=0.1)
    plt.title('Drivers and SWB over time', fontsize=16)
    plt.xlabel('Iteration', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend()

    # Rotate x-axis labels to be vertical and set font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save and close figure
    plt.savefig(f"figures/single_{title_add}.pdf", dpi=300)
    plt.clf()
    plt.close()
    return
    
def SWB_gif(model, output, iterations, fps, name="test", xlabel="", ylabel="", xlim=[0, 10], ylim=[0, 10]):
    """
    Create a GIF of SWB over iterations.

    Parameters:
    - model: The model object.
    - output: The output data from the model.
    - iterations: The number of iterations.
    - fps: Frames per second for the GIF.
    - name: The name of the GIF file.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - xlim: Limits for the x-axis.
    - ylim: Limits for the y-axis.

    This function extracts data and generates a GIF showing SWB over time.
    """
    data = extract_data(output, 1)
    gif(data, iterations, fps, name=name, xlabel=xlabel, ylabel=ylabel, xlim=xlim)

def gif(data, frames, fps, name="test", xlabel="", ylabel="", xlim=[0, 10], ylim=[0, 10]):
    """
    Create a GIF from data.

    Parameters:
    - data: The data to visualize.
    - frames: Number of frames in the GIF.
    - fps: Frames per second for the GIF.
    - name: The name of the GIF file.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - xlim: Limits for the x-axis.
    - ylim: Limits for the y-axis.

    This function generates and saves a GIF animation.
    """
    number_of_frames = 100

    fig = plt.figure()
    hist = plt.hist(data[1])
    writergif = animation.PillowWriter(fps=fps)
    anim = animation.FuncAnimation(fig, update_hist, frames, fargs=(data, xlabel, ylabel, xlim) )
    anim.save(f'animations/{name}.gif',writer=writergif)

def update_hist(num, data, xlabel, ylabel, xlim):
    """
    Update histogram for each frame in the GIF.

    Parameters:
    - num: Current frame number.
    - data: The data to visualize.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - xlim: Limits for the x-axis.

    This function updates the histogram for the animation.
    """
    plt.cla()
    plt.hist(data[num], range=xlim)
    plt.title(f"Iteration: {num}")
    # plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def plot_var(data, intervention_timesteps, int_var, title_add=""):
    """
    Plot variance of SWB over time.

    Parameters:
    - data: The data to plot.
    - intervention_timesteps: Timesteps where interventions occur.
    - int_var: Variables indicating the type of intervention.
    - title_add: Additional title text for the plot.

    This function generates a line plot with shaded standard deviation areas.
    """
    added_to_legend = []
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    plt.plot(np.mean(data, axis=0))
    plt.fill_between(np.linspace(0, len(mean), len(mean)), mean-std, mean+std, alpha=0.1)

    for intervention_index, intervention in enumerate(intervention_timesteps):
        if int_var[intervention_index] == "fin":
            line_color = "green"
        elif int_var[intervention_index] == "nonfin":
            line_color = "blue"
        if int_var[intervention_index] not in added_to_legend:
            plt.axvline(x=intervention, color=line_color, linestyle='--', label=int_var[intervention_index] + " shock")
            added_to_legend.append(int_var[intervention_index])
        else:
            plt.axvline(x=intervention, color=line_color, linestyle='--')
    plt.legend()
    plt.title(f"Variance in SWB over time")
    plt.xlabel("Iteration")
    plt.ylabel(f"Variance")
    plt.savefig(f"figures/var_avg{title_add}.pdf", dpi=300)
    plt.clf()
    plt.close()


def two_var_heatmap(data, baseline, params, samples_1, samples_2, title_add="", per_person=False):
    """
    Create a heatmap for two variables.

    Parameters:
    - data: The data to visualize.
    - baseline: Baseline data for comparison.
    - params: The parameters being varied.
    - samples_1: Sample values for the first parameter.
    - samples_2: Sample values for the second parameter.
    - title_add: Additional title text for the plot.
    - per_person: Flag to indicate if the plot is per person.

    This function generates a heatmap and saves it as a PDF.
    """
    data = np.mean(data, axis=2)

    # plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, cmap="Wistia", fmt=".2f")
    if per_person:
        plt.title('Proportion of agents for which SWB changed after intervention', fontsize=16)
    else:
        plt.title('Mean SWB after intervention', fontsize=16)
    plt.xlabel(f'{params[1]}', fontsize=14)
    plt.ylabel(f'Mean {params[0]}', fontsize=14)
    # plt.xlabel("Intervention factor")
    # plt.ylabel("Amount of interventions per habituation period")
    plt.xticks(ticks=np.arange(len(samples_2)) + 0.5, labels=[f'{x:.2f}' for x in samples_2], fontsize=12)
    plt.yticks(ticks=np.arange(len(samples_1)) + 0.5, labels=[f'{x:.2f}' for x in samples_1], fontsize=12)
    plt.savefig(f"figures/heatmap{title_add}.pdf", dpi=300)
    plt.clf()
    plt.close()

    if not per_person:
        baseline = np.mean(baseline, axis=2)
        sns.heatmap(data-baseline, annot=True, cmap="Wistia", fmt=".2f")
        plt.title('Difference in SWB after compared to before interventions', fontsize=16)
        plt.xlabel(f'{params[1]}', fontsize=14)
        plt.ylabel(f'Mean {params[0]}', fontsize=14)
        # plt.xlabel("Intervention factor", fontsize=14)
        # plt.ylabel("Amount of interventions per habituation period", fontsize=14)
        plt.xticks(ticks=np.arange(len(samples_2)) + 0.5, labels=[f'{x:.2f}' for x in samples_2], fontsize=12)
        plt.yticks(ticks=np.arange(len(samples_1)) + 0.5, labels=[f'{x:.2f}' for x in samples_1], fontsize=12)
        plt.savefig(f"figures/heatmap_diff{title_add}.pdf", dpi=300)
        plt.clf()
        plt.close()
    return

def hist_plot(data, title_add=""):
    """
    Plot histogram of SWB data.

    Parameters:
    - data: The data to plot.
    - title_add: Additional title text for the plot.

    This function generates a histogram of SWB data and saves it as a PDF.
    """
    # Create the histogram
    counts, _, patches = plt.hist(data, bins=15, edgecolor='black')

    # Apply the Wistia colormap
    colormap = plt.cm.get_cmap('Wistia')

    # Normalize the counts to the range [0, 1] for the colormap
    norm = plt.Normalize(counts.min(), counts.max())

    # Apply the colormap to each patch (bin)
    for count, patch in zip(counts, patches):
        color = colormap(norm(count))
        patch.set_facecolor(color)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])

    plt.title('Distribution of SWB over the network', fontsize=16)
    plt.xlabel('SWB', fontsize=14)
    plt.ylabel('Amount of agents', fontsize=14)

    # Rotate x-axis labels to be vertical and set font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(f"figures/hist{title_add}.pdf", dpi=300)
    plt.clf()
    plt.close()
    return

def degree_plot(df, title_add=""):
    """
    Plot the effect of degree on SWB.

    Parameters:
    - df: The DataFrame containing degree and SWB data.
    - title_add: Additional title text for the plot.

    This function generates a line plot with shaded standard deviation areas showing the effect of degree on SWB.
    """
    means = df.groupby("degree").mean()["SWB"]
    stds = df.groupby("degree").std()["SWB"]
    plt.plot(means)
    plt.fill_between(np.linspace(0, len(means)-1, len(means)), means-stds, means+stds, alpha=0.1)

    plt.title('Degree effects', fontsize=16)
    plt.xlabel('Degree', fontsize=14)
    plt.ylabel('SWB', fontsize=14)

    # Rotate x-axis labels to be vertical and set font size
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(f"figures/degree_SWB{title_add}.pdf", dpi=300)
    plt.clf()
    plt.close()
    return