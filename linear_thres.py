import numpy as np
import pandas as pd
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from bokeh.io import output_notebook, show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend
import matplotlib.pyplot as plt
from tqdm import tqdm

## Global Constants
N = 1000 #1000
#P_LST = np.linspace(0, 0.5, 101)[1:]
P = 0.01
#THRES_LST = np.random.uniform(0, 1, size = N) #0.005
CUTOFF = (N * P - 1 + (1 - P) ** N) / (N * P)
THRES_UP_LST = np.linspace(0.4, 1, 7) #101
FRAC_INFECTED = round(1 / np.log(N) ** 2, 2) #0.05
NUM_ITR = 20
NUM_TRIALS = 30 #200

def single_trial_thres(THRES_UP, get_graph = False):
    # Model Configuration
    config = mc.Configuration()
    config.add_model_parameter("fraction_infected", FRAC_INFECTED)
    # Setting node parameters
    THRES_LST = np.random.uniform(0, THRES_UP, size = N)
    for i in range(N):
        config.add_node_configuration("threshold", i, THRES_LST[i])
    # Network topology
    g = nx.erdos_renyi_graph(N, P)
    # Model selection
    model = ep.ThresholdModel(g)
    model.set_initial_status(config)
    # Simulation execution
    iterations = model.iteration_bunch(NUM_ITR)
    trends = model.build_trends(iterations)
    frac_infected = trends[0]["trends"]["node_count"][1][-1] / N
    snapshot = {}
    if get_graph:
        for i in range(NUM_ITR):
            dct = iterations[i]["status"]
            for key in dct:
                snapshot[key] = dct[key]
    return frac_infected, (g, snapshot)

def single_trial_degree(THRES_UP, get_graph = False):
    # Model Configuration
    config = mc.Configuration()
    config.add_model_parameter("fraction_infected", FRAC_INFECTED)
    # Setting node parameters
    THRES_LST = np.random.uniform(0, THRES_UP, size = N)
    for i in range(N):
        config.add_node_configuration("threshold", i, THRES_LST[i])
    # Network topology
    g = nx.erdos_renyi_graph(N, P)
    # Model selection
    model = ep.ThresholdModel(g)
    model.set_initial_status(config)
    # Simulation execution
    iterations = model.iteration_bunch(NUM_ITR)
    ## Collect infected nodes with their corresponding degrees
    nodes_infected_status = [0] * N
    nodes_degrees = [g.degree(x) for x in range(N)]
    for itr in iterations:
        dct = itr["status"]
        for key in dct:
            if dct[key] == 1:
                nodes_infected_status[key] = 1
    snapshot = {}
    if get_graph:
        for i in range(NUM_ITR):
            dct = iterations[i]["status"]
            for key in dct:
                snapshot[key] = dct[key]
    df = pd.DataFrame.from_dict({"Degree": nodes_degrees, "FracInfected": nodes_infected_status})
    df = df.groupby("Degree").mean().reset_index()
    degree_frac_infected = np.zeros(N)
    max_deg = 0
    for i in range(df.shape[0]):
        deg = int(df.iloc[i]["Degree"])
        frac_infected = df.iloc[i]["FracInfected"]
        max_deg = max(max_deg, deg)
        degree_frac_infected[deg] = frac_infected
    return degree_frac_infected, max_deg, (g, snapshot)

#P = 0.1
#frac_infected, (g, snapshot) = single_trial(P, get_graph = True)
#color_map = []
#for node in g:
#    if snapshot[node] == 1:
#        color_map.append("#ffc3d7")
#    else:
#        color_map.append("#badaff")
#cent=nx.edge_betweenness_centrality(g)
#node_pos=nx.spring_layout(g)
#nx.draw_networkx_nodes(g, pos = node_pos, node_color = color_map, node_size = 50)
#nx.draw_networkx_edges(g, pos = node_pos, edgelist = cent, alpha = 0.1)
#plt.savefig(f"Plots/snapshot_p={P}_rho={FRAC_INFECTED}.png")
#plt.clf()
#plt.close()

## Get total infected fraction per threshold
#frac_infected_lst = []
#upper_lst = []
#lower_lst = []
#for THRES_UP in tqdm(THRES_UP_LST):
#    frac_infected_trials = np.zeros(NUM_TRIALS)
#    for trial in tqdm(range(NUM_TRIALS), leave = False):
#        frac_infected, _ = single_trial_thres(THRES_UP)
#        frac_infected_trials[trial] = frac_infected
#    frac_infected_lst.append(np.mean(frac_infected_trials))
#    upper_lst.append(np.quantile(frac_infected_trials, 0.975))
#    lower_lst.append(np.quantile(frac_infected_trials, 0.025))
#
#plt.plot(THRES_UP_LST, frac_infected_lst)
#plt.fill_between(THRES_UP_LST, lower_lst, upper_lst, alpha = 0.1)
#plt.axvline(x = CUTOFF, color = "red")
#plt.axvline(x = CUTOFF / 2, color = "green")
#plt.xlabel("Maximum Threshold")
#plt.ylabel("Fraction of Infected")
#plt.title(f"rho = {FRAC_INFECTED}, p = {P}")
#plt.savefig(f"Plots/linearThres_rho={FRAC_INFECTED}_p={P}.png")
#plt.clf()
#plt.close()

## Get number of infected nodes per degree per threshold
frac_infected_lst = []
upper_lst = []
lower_lst = []
for THRES_UP in tqdm(THRES_UP_LST):
    frac_infected_trials = np.zeros((NUM_TRIALS, N))
    minmax_deg = N
    for trial in tqdm(range(NUM_TRIALS), leave = False):
        degree_frac_infected, max_deg, _ = single_trial_degree(THRES_UP)
        frac_infected_trials[trial,:] = degree_frac_infected
        minmax_deg = min(minmax_deg, max_deg)
    frac_infected_lst = np.mean(frac_infected_trials, axis = 0)[:(minmax_deg + 1)]
    upper_lst = np.quantile(frac_infected_trials, 0.975, axis = 0)[:(minmax_deg + 1)]
    lower_lst = np.quantile(frac_infected_trials, 0.025, axis = 0)[:(minmax_deg + 1)]
    degree_lst = np.arange(minmax_deg + 1)

    plt.plot(degree_lst, frac_infected_lst)
    plt.fill_between(degree_lst, lower_lst, upper_lst, alpha = 0.1)
    plt.xlabel("Node Degree")
    plt.ylabel("Fraction of Infected")
    plt.title(f"1/m = {round(THRES_UP, 2)}, Cutoff = {round(CUTOFF, 2)}\nrho = {FRAC_INFECTED}, p = {P}")
    plt.savefig(f"Plots/degree_maxthres={round(THRES_UP, 2)}_rho={FRAC_INFECTED}_p={P}.png")
    plt.clf()
    plt.close()

## Debugging Region
#single_trial(0.8)
