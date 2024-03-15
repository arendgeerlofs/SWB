import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import timeit
from parameters import network_parameters

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
    SWB_scores = [[output["states"][a][0][0]] for a in output["states"]]

    fin_scores = [[output["states"][a][0][10]] for a in output["states"]]
    expectation_scores = [[output["states"][a][0][13]] for a in output["states"]]
    RFC = [[output["states"][a][0][14]] for a in output["states"]]

    # Plot SWB scores over time
    # TODO change to averages
    # print(SWB_scores)
    plt.plot(SWB_scores[2:])
    plt.savefig("figures/SWB")
    plt.clf()   # Clear figure
    plt.plot(fin_scores[2:])
    plt.plot(expectation_scores[2:])
    plt.savefig("figures/fin")
    plt.clf()
    plt.plot(RFC[2:])
    plt.plot(expectation_scores[2:])
    plt.savefig("figures/RFC")

def extract_data(output, state_number):
    data = np.zeros((len(output["states"])+1, network_parameters["N"]))
    for timestep in output["states"]:
        for person, _ in enumerate(output["states"][timestep]):
            data[timestep][person] = output["states"][timestep][person][state_number]
    return data

def SWB_gif(output, iterations, fps, name="test", xlabel="", ylabel="", xlim=[0, 10], ylim=[0, 10]):
    # Get SWB data
    data = extract_data(output, 0)
    gif(data, iterations, fps, name=name, xlabel=xlabel, ylabel=ylabel, xlim=xlim)

def gif(data, frames, fps, name="test", xlabel="", ylabel="", xlim=[0, 10], ylim=[0, 10]):
    number_of_frames = 100

    fig = plt.figure()
    hist = plt.hist(data[0])
    writergif = animation.PillowWriter(fps=fps)
    anim = animation.FuncAnimation(fig, update_hist, frames, fargs=(data, xlabel, ylabel, xlim) )
    anim.save(f'animations/{name}.gif',writer=writergif)

def update_hist(num, data, xlabel, ylabel, xlim):
    plt.cla()
    plt.hist(data[num])
    plt.title(f"Iteration: {num}")
    plt.xlim(xlim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)