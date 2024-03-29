import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import timeit
from parameters import network_parameters
from functions import extract_data

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

def plot(output):
    # Plot data
    # Print SWB scores over time of person 0
    SWB_scores = [[output["states"][a][0][1]] for a in output["states"]]
    SWB_exp = [[output["states"][a][0][2]] for a in output["states"]]
    SWB_norm = [[output["states"][a][0][0]] for a in output["states"]]

    fin_scores = [[output["states"][a][0][5]] for a in output["states"]]
    expectation_scores = [[output["states"][a][0][6]] for a in output["states"]]
    RFC = [[output["states"][a][0][7]] for a in output["states"]]
    RFC_exp = [[output["states"][a][0][8]] for a in output["states"]]

    # Plot SWB scores over time
    # TODO change to averages
    # print(SWB_scores)
    plt.plot(SWB_scores)
    plt.plot(SWB_exp)
    plt.plot(SWB_norm, 'g')
    plt.savefig("figures/SWB")
    plt.clf()   # Clear figure
    plt.plot(fin_scores)
    plt.plot(expectation_scores)
    plt.savefig("figures/fin")
    plt.clf()
    plt.plot(RFC)
    plt.plot(RFC_exp)
    plt.savefig("figures/RFC")

def SWB_gif(output, iterations, fps, name="test", xlabel="", ylabel="", xlim=[0, 10], ylim=[0, 10]):
    # Get SWB data
    data = extract_data(output, 1)
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