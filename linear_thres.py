import numpy as np
import pandas as pd
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from bokeh.io import output_notebook, show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

## Global Constants
N = 1000 #100
#P_LST = np.linspace(0, 0.5, 101)[1:]
Z = 3
P = Z / N
#THRES_LST = np.random.uniform(0, 1, size = N) #0.005
THRES_UP_LST = [0.3, 0.6, 0.9] #np.linspace(0, 1, 21) #101
FRAC_INFECTED = round(1 / np.log(N) ** 2, 2) #0.05
NUM_ITR = 80
NUM_TRIALS = 200 #30 #200
D = 1
BETA = 2/3
W = D * (N / np.arange(1, N + 1)) ** BETA
GRAPH_TYPE = "Chung-Lu" # Gnp, Chung-Lu
if GRAPH_TYPE == "Gnp":
    CUTOFF = (N * P - 1 + (1 - P) ** N) / (N * P)
else:
    w_tot = np.sum(W)
    Z = np.mean(W * (1 - W / w_tot))
    CUTOFF = (Z - 1) / Z
print(CUTOFF)

fig = plt.figure()

def draw_graph(i, g, iterations, cent, node_pos, snapshot, thres):
    plt.clf()
    color_map = []
    dct = iterations[i]["status"]
    for key in dct:
        snapshot[key] = dct[key]
    num_nodes = 0
    infected_nodes = 0
    for node in g:
        if node in snapshot and snapshot[node] == 1:
            color_map.append("#ffc3d7")
            infected_nodes += 1
        else:
            color_map.append("#badaff")
        num_nodes += 1
    frac_infected = infected_nodes / num_nodes
    nx.draw_networkx_nodes(g, pos = node_pos, node_color = color_map, node_size = 50)
    nx.draw_networkx_edges(g, pos = node_pos, edgelist = cent, alpha = 0.1)
#    plt.title(f"Threshold: {thres} - Iteration #{i+1} Infected {(frac_infected * 100):.2f}%\nCutoff: {CUTOFF:.2f}")
    if i in [0, 4, 9]:
        plt.savefig(f"Snapshots/type={GRAPH_TYPE.lower()}_thres={thres}_itr={i+1}.png")

def single_trial_thres(THRES_UP, get_graph = False):
    # Model Configuration
    config = mc.Configuration()
    config.add_model_parameter("fraction_infected", FRAC_INFECTED)
    # Setting node parameters
    THRES_LST = np.random.uniform(0, THRES_UP, size = N)
    for i in range(N):
        config.add_node_configuration("threshold", i, THRES_LST[i])
    # Network topology
    if GRAPH_TYPE == "Gnp":
        g = nx.erdos_renyi_graph(N, P)
    else:
        g = nx.expected_degree_graph(W, selfloops = False)
    largest_cc = max(nx.connected_components(g), key = len)
    # Model selection
    model = ep.ThresholdModel(g)
    model.set_initial_status(config)
    # Simulation execution
    iterations = model.iteration_bunch(NUM_ITR)
    trends = model.build_trends(iterations)
#    frac_infected = trends[0]["trends"]["node_count"][1][-1] / N
    snapshot = {}
    num_infected = 0
    num_nodes = 0
    if True: #get_graph:
        for i in range(NUM_ITR):
            dct = iterations[i]["status"]
            for key in dct:
                snapshot[key] = dct[key]
    for key in snapshot:
        if key in largest_cc:
            num_infected += snapshot[key]
            num_nodes += 1
    frac_infected = num_infected / num_nodes
    return frac_infected, (g, snapshot, iterations)

def single_trial_degree(THRES_UP, get_graph = False):
    # Model Configuration
    config = mc.Configuration()
    config.add_model_parameter("fraction_infected", FRAC_INFECTED)
    # Setting node parameters
    THRES_LST = np.random.uniform(0, THRES_UP, size = N)
    for i in range(N):
        config.add_node_configuration("threshold", i, THRES_LST[i])
    # Network topology
    if GRAPH_TYPE == "Gnp":
        g = nx.erdos_renyi_graph(N, P)
    else:
        g = nx.expected_degree_graph(W, selfloops = False)
    largest_cc = max(nx.connected_components(g), key = len)
    # Model selection
    model = ep.ThresholdModel(g)
    model.set_initial_status(config)
    # Simulation execution
    iterations = model.iteration_bunch(NUM_ITR)
    ## Collect infected nodes with their corresponding degrees
    nodes_infected_status = [0] * N
    nodes_degrees = [g.degree(x) for x in range(N)]
    is_largest_component = [1 if x in largest_cc else 0 for x in range(N)]
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
    df = pd.DataFrame.from_dict({"Degree": nodes_degrees, "FracInfected": nodes_infected_status, "InLargestCC": is_largest_component})
    ## Filter the largest connected component
    df = df[df["InLargestCC"] == 1]
    df_full = df.copy()
    df["Count"] = 1
    df_deg = df[["Degree", "Count"]].groupby("Degree").sum().reset_index()
    df_deg = df_deg[df_deg["Count"] >= 1]
    max_deg = df_deg["Degree"].max()
    frac_infected_all = df_full[df_full["Degree"] > 0][["FracInfected"]].mean()
    df = df.groupby("Degree").mean().reset_index()
    degree_frac_infected = np.empty(N)
#    max_deg = 0
    for i in range(df.shape[0]):
        deg = int(df.iloc[i]["Degree"])
        frac_infected = df.iloc[i]["FracInfected"]
#        max_deg = max(max_deg, deg)
        degree_frac_infected[deg] = frac_infected
    return degree_frac_infected, max_deg, frac_infected_all, (g, snapshot)

## Animation
#THRES_UP = 0.9 # 0.3, 0.6, 0.9
#frac_infected, (g, snapshot, iterations) = single_trial_thres(THRES_UP, get_graph = True)
#largest_cc = max(nx.connected_components(g), key = len)
#g = g.subgraph(largest_cc)
#cent=nx.edge_betweenness_centrality(g)
#node_pos=nx.spring_layout(g)
#snapshot = {}
#anim = FuncAnimation(fig, draw_graph, frames = len(iterations), interval = 1000, fargs=(g, iterations, cent, node_pos, snapshot, THRES_UP), repeat = True)
#plt.show()

## Single Snapshot
#largest_cc = max(nx.connected_components(g), key = len)
#g = g.subgraph(largest_cc)
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
#plt.savefig(f"Plots/snapshot_n={N}_thres={THRES_UP}_z={Z}.png")
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
##plt.axvline(x = CUTOFF / 2, color = "green")
#plt.axhline(y = 1, color = "black")
#plt.xlabel("Maximum Threshold")
#plt.ylabel("Fraction of Infected")
##plt.title(f"rho = {FRAC_INFECTED}, p = {P}")
#if GRAPH_TYPE == "Gnp":
#    plt.savefig(f"FracInfected/frac-infected_type=gnp_n={N}_p={P}.png")
#else:
#    plt.savefig(f"FracInfected/frac-infected_type=chung-lu_d={D}_beta={round(BETA, 2)}.png")
#plt.clf()
#plt.close()

## Get number of infected nodes per degree per threshold
frac_infected_lst = []
upper_lst = []
lower_lst = []
for THRES_UP in tqdm(THRES_UP_LST):
    frac_infected_trials = np.zeros((NUM_TRIALS, N))
    minmax_deg = N
    frac_infected_all_trials = np.zeros(NUM_TRIALS)
    for trial in tqdm(range(NUM_TRIALS), leave = False):
        degree_frac_infected, max_deg, frac_infected_all, _ = single_trial_degree(THRES_UP)
        frac_infected_trials[trial,:] = degree_frac_infected
        frac_infected_all_trials[trial] = frac_infected_all
        minmax_deg = min(minmax_deg, max_deg)
    frac_infected_lst = np.nanmean(frac_infected_trials, axis = 0)[1:(minmax_deg + 1)]
    upper_lst = np.nanquantile(frac_infected_trials, 0.975, axis = 0)[1:(minmax_deg + 1)]
    lower_lst = np.nanquantile(frac_infected_trials, 0.025, axis = 0)[1:(minmax_deg + 1)]
    degree_lst = np.arange(1, minmax_deg + 1)
    frac_infected_all = np.mean(frac_infected_all_trials)

    plt.plot(degree_lst, frac_infected_lst)
    plt.fill_between(degree_lst, lower_lst, upper_lst, alpha = 0.1)
#    plt.axvline(x = 1, color = "red", label = "degree = 1")
    plt.axhline(y = frac_infected_all, color = "green", label = "Pop Avg Infected")
    plt.xlabel("Node Degree")
    plt.ylabel("Fraction of Infected")
    plt.ylim(0, 1.1)
    plt.legend()
#    plt.title(f"1/m = {round(THRES_UP, 2)}, Cutoff = {round(CUTOFF, 2)}\nrho = {FRAC_INFECTED}, p = {P}")
    if GRAPH_TYPE == "Gnp":
        plt.savefig(f"Degree/degree_type=gnp_n={N}_p={P}_maxthres={round(THRES_UP, 2)}.png")
    else:
        plt.savefig(f"Degree/degree_type=chung-lu_d={D}_beta={round(BETA, 2)}_maxthres={round(THRES_UP, 2)}_full.png")
    plt.clf()
    plt.close()

## Debugging Region
#single_trial(0.8)
#frac_infected, (g, snapshot) = single_trial_thres(0.8, get_graph = True)
#print(frac_infected)
#print(snapshot)
