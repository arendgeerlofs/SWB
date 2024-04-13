import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import timeit
import pandas as pd
from functions import extract_data
from initialise import init_states

def visualise(model, output): 
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

def plot(output, columns_to_plot= ["SWB", "SWB_exp", "financial", "fin_exp", "nonfin", "nonfin_exp"], plot_from=0):
    arrays = [output['states'][key] for key in output['states']]
    data = np.stack(arrays, axis=0)
    pop_avg = np.mean(data, axis=1)
    df_pop_avg = pd.DataFrame(pop_avg, columns=list(init_states.keys()))
    # df_pop_avg[columns_to_plot].plot()

    ax = df_pop_avg[columns_to_plot].plot(marker='.', label=columns_to_plot)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Drivers and SWB over time')
    ax.legend()

    plt.savefig("avg")  


def plot_for_one(output, plot_from=0):
    # Plot data
    # Print SWB scores over time of person 0
    SWB_scores = [[output["states"][a][0][1]] for a in output["states"]]
    SWB_exp = [[output["states"][a][0][2]] for a in output["states"]]
    SWB_norm = [[output["states"][a][0][0]] for a in output["states"]]

    fin_scores = [[output["states"][a][0][8]] for a in output["states"]]
    expectation_scores = [[output["states"][a][0][9]] for a in output["states"]]
    nonfin_scores = [[output["states"][a][0][10]] for a in output["states"]]
    nonfin_expectation_scores = [[output["states"][a][0][11]] for a in output["states"]]
    fin_sens = [[output["states"][a][0][12]] for a in output["states"]]
    nonfin_sens = [[output["states"][a][0][13]] for a in output["states"]]
    RFC = [[output["states"][a][0][14]] for a in output["states"]]
    RFC_exp = [[output["states"][a][0][15]] for a in output["states"]]

    # Plot SWB scores over time
    # TODO change to averages
    # print(SWB_scores)
    plt.plot(SWB_scores[plot_from:], 'b.-')
    plt.plot(SWB_exp[plot_from:], 'r.-')
    plt.plot(SWB_norm[plot_from:], 'g.-')
    # plt.ylim(0, 10)
    plt.savefig("figures/SWB")
    plt.clf()   # Clear figure
    plt.plot(fin_scores[plot_from:], 'b.-')
    plt.plot(expectation_scores[plot_from:], 'r.-')
    # plt.ylim(0, 10)
    plt.savefig("figures/fin")
    plt.clf()
    plt.plot(nonfin_scores[plot_from:], 'b.-')
    plt.plot(nonfin_expectation_scores[plot_from:], 'r.-')
    # plt.ylim(0, 10)
    plt.savefig("figures/nonfin")
    plt.clf()
    plt.plot(fin_sens[plot_from:], 'b.-')
    plt.plot(nonfin_sens[plot_from:], 'r.-')
    plt.savefig("figures/sens")
    plt.clf()
    plt.plot(RFC[plot_from:], 'b.-')
    plt.plot(RFC_exp[plot_from:], 'r.-')
    plt.ylim(0, 10)
    plt.savefig("figures/RFC")

def SWB_gif(model, output, iterations, fps, name="test", xlabel="", ylabel="", xlim=[0, 10], ylim=[0, 10]):
    # Get SWB data
    data = extract_data(model.constants["N"], output, 1)
    gif(data, iterations, fps, name=name, xlabel=xlabel, ylabel=ylabel, xlim=xlim)

def gif(data, frames, fps, name="test", xlabel="", ylabel="", xlim=[0, 10], ylim=[0, 10]):
    number_of_frames = 100

    fig = plt.figure()
    hist = plt.hist(data[1])
    writergif = animation.PillowWriter(fps=fps)
    anim = animation.FuncAnimation(fig, update_hist, frames, fargs=(data, xlabel, ylabel, xlim) )
    anim.save(f'animations/{name}.gif',writer=writergif)

def update_hist(num, data, xlabel, ylabel, xlim):
    plt.cla()
    plt.hist(data[num], range=xlim)
    plt.title(f"Iteration: {num}")
    # plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)