import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
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

def plot_avg(output, columns_to_plot= ["SWB", "SWB_exp", "financial", "fin_exp", "nonfin", "nonfin_exp", "soc_cap", "soc_cap_exp"], plot_from=0, name_addition="avgs"):
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
    plt.savefig(f"figures/{name_addition}")
    ax = df_pop_avg[columns_to_plot].plot(marker='.', label=columns_to_plot, ylim=(0, 10))
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title('Drivers and SWB over time')
    ax.legend()
    plt.savefig(f"figures/ylim_{name_addition}")


def plot_for_one(output, plot_from=0, name_addition="", states_to_plot=["SWB", "fin", "nonfin", "sens", "RFC", "soc_cap"]):
    # Plot data
    # Print SWB scores over time of person 0
    if "SWB" in states_to_plot:
        SWB_scores = [[output["states"][a][0][1]] for a in output["states"]]
        SWB_exp = [[output["states"][a][0][2]] for a in output["states"]]
        SWB_norm = [[output["states"][a][0][0]] for a in output["states"]]
        plt.plot(SWB_scores[plot_from:], 'b.-', label="Actual SWB")
        plt.plot(SWB_exp[plot_from:], 'r.-', label="expection SWB")
        plt.plot(SWB_norm[plot_from:], 'g.-', label="Norm SWB")
        plt.legend()
        plt.savefig(f"figures/SWB_{name_addition}")
        plt.clf()

    if "fin" in states_to_plot:
        fin_scores = [[output["states"][a][0][8]] for a in output["states"]]
        expectation_scores = [[output["states"][a][0][9]] for a in output["states"]]
        plt.plot(fin_scores[plot_from:], 'b.-', label="Economic")
        plt.plot(expectation_scores[plot_from:], 'r.-', label="Expection")
        plt.legend()
        # plt.ylim(0, 10)
        plt.savefig(f"figures/fin_{name_addition}")
        plt.clf()
    
    if "nonfin" in states_to_plot:
        nonfin_scores = [[output["states"][a][0][10]] for a in output["states"]]
        nonfin_expectation_scores = [[output["states"][a][0][11]] for a in output["states"]]
        plt.plot(nonfin_scores[plot_from:], 'b.-', label="Non-Economic")
        plt.plot(nonfin_expectation_scores[plot_from:], 'r.-', label="Expectation")
        plt.legend()
        # plt.ylim(0, 10)
        plt.savefig(f"figures/nonfin_{name_addition}")
        plt.clf()
    
    if "sens" in states_to_plot:
        fin_sens = [[output["states"][a][0][12]] for a in output["states"]]
        nonfin_sens = [[output["states"][a][0][13]] for a in output["states"]]
        plt.plot(fin_sens[plot_from:], 'b.-', label="Economic sensitivity")
        plt.plot(nonfin_sens[plot_from:], 'r.-', label="Non-Economic sensitivity")
        plt.legend()
        plt.savefig(f"figures/sens_{name_addition}")
        plt.clf()
    
    if "RFC" in states_to_plot:
        RFC = [[output["states"][a][0][14]] for a in output["states"]]
        RFC_exp = [[output["states"][a][0][15]] for a in output["states"]]
        plt.plot(RFC[plot_from:], 'b.-', label="RFC")
        plt.plot(RFC_exp[plot_from:], 'r.-', label="Expectation")
        plt.legend()
        plt.ylim(0, 10)
        plt.savefig(f"figures/RFC_{name_addition}")
        plt.clf()

    if "soc_cap" in states_to_plot:
        soc_cap = [[output["states"][a][0][16]] for a in output["states"]]
        soc_cap_exp = [[output["states"][a][0][17]] for a in output["states"]]
        plt.plot(soc_cap[plot_from:], 'b.-', label="Social Capital")
        plt.plot(soc_cap_exp[plot_from:], 'r.-', label="Expectation")
        plt.legend()
        plt.savefig(f"figures/soc_cap_{name_addition}")
        plt.clf()
    
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

# def stoch_var_change_plot(constants, its=100, runs=10, samples=10, var_to_change="fin_event_prob", bounds=[0, 0.5]):
#     data = np.empty((samples, runs, constants["N"]))
#     for i, sample_value in enumerate(np.linspace(bounds[0], bounds[1], samples)):
#         new_constants = constants
#         new_constants[var_to_change] = sample_value
#         for j in range(runs):
#             output = exec(new_constants, its)
#             run_data = 
