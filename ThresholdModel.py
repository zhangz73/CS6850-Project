import sys
import math
import networkx as nx
import numpy as np
import pandas as pd
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from tqdm import tqdm

GRAPH_NAME = sys.argv[1]

data = pd.read_csv(f"{GRAPH_NAME}.txt", sep="\t").dropna()
# data = data.astype(np.int8)
all_nodes = list(set(list(data["FromNodeId"]) + list(data["ToNodeId"])))
# nodes_map = {}
# for i in range(len(all_nodes)):
#     nodes_map[all_nodes[i]] = i
# for i in range(data.shape[0]):
#     data.loc[i, "FromNodeId"] = nodes_map[data.iloc[i]["FromNodeId"]]
#     data.loc[i, "ToNodeId"] = nodes_map[data.iloc[i]["ToNodeId"]]
GRAPH = nx.from_pandas_edgelist(data, 'FromNodeId', 'ToNodeId')
N = len(GRAPH.nodes)

# Use same parameters as Zhanhao
THRES_UP_LST = np.linspace(0, 1, 11) #[0.3, 0.6, 0.9] #
FRAC_INFECTED = round(1 / np.log(N) ** 2, 2)
NUM_ITR = 10 # 80
NUM_TRIALS = 10 # 100
n_cpu = 11
Z = np.sum([val for (node, val) in GRAPH.degree()]) / N
CUTOFF = (Z - 1) / Z
print(Z, CUTOFF)

def single_trial_thres(THRES_UP, g):
    # Model Configuration
    config = mc.Configuration()
    config.add_model_parameter("fraction_infected", FRAC_INFECTED)
    # Setting node parameters
    THRES_LST = np.random.uniform(0, THRES_UP, size = N)
    for i in range(N):
        config.add_node_configuration("threshold", i, THRES_LST[i])
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

def single_trial_degree(THRES_UP, g, get_graph = False):
    # Model Configuration
    config = mc.Configuration()
    config.add_model_parameter("fraction_infected", FRAC_INFECTED)
    # Setting node parameters
    THRES_LST = np.random.uniform(0, THRES_UP, size = N)
    all_nodes = list(g.nodes)
    for i in range(N):
        config.add_node_configuration("threshold", i, THRES_LST[i])
    # Network topology
#     if GRAPH_TYPE == "Gnp":
#         g = nx.erdos_renyi_graph(N, P)
#     else:
#         g = nx.expected_degree_graph(W, selfloops = False)
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
    df_deg = df_deg[df_deg["Count"] >= N * 0.003]
    max_deg = df_deg["Degree"].max()
    frac_infected_all = df_full[df_full["Degree"] > 0]["FracInfected"].mean()
    df = df.groupby("Degree").mean().reset_index()
    degree_frac_infected = np.empty(N)
#    max_deg = 0
    for i in range(df.shape[0]):
        deg = int(df.iloc[i]["Degree"])
        frac_infected = df.iloc[i]["FracInfected"]
#        max_deg = max(max_deg, deg)
        degree_frac_infected[deg] = frac_infected
    return degree_frac_infected, max_deg, frac_infected_all, (g, snapshot)

def thres_trials_single(thres_up_lst):
    frac_infected_lst = []
    upper_lst = []
    lower_lst = []
    for THRES_UP in tqdm(thres_up_lst):
        frac_infected_trials = np.zeros(NUM_TRIALS)
        for trial in tqdm(range(NUM_TRIALS), leave = False):
            frac_infected, _ = single_trial_thres(THRES_UP, GRAPH)
            frac_infected_trials[trial] = frac_infected
        frac_infected_lst.append(np.mean(frac_infected_trials))
        upper_lst.append(np.quantile(frac_infected_trials, 0.975))
        lower_lst.append(np.quantile(frac_infected_trials, 0.025))
    return frac_infected_lst, upper_lst, lower_lst

def degree_trials_single(thres_up_lst):
    frac_infected_lst = []
    upper_lst = []
    lower_lst = []
    degree_lst = []
    frac_infected_all_lst = []
    for THRES_UP in tqdm(thres_up_lst):
        frac_infected_trials = np.zeros((NUM_TRIALS, N))
        minmax_deg = N
        frac_infected_all_trials = np.zeros(NUM_TRIALS)
        for trial in tqdm(range(NUM_TRIALS), leave = False):
            degree_frac_infected, max_deg, frac_infected_all, _ = single_trial_degree(THRES_UP, GRAPH)
            frac_infected_trials[trial,:] = degree_frac_infected
            frac_infected_all_trials[trial] = frac_infected_all
            minmax_deg = min(minmax_deg, max_deg)
        frac_infected_lst.append(np.nanmean(frac_infected_trials, axis = 0)[1:(minmax_deg + 1)])
        upper_lst.append(np.nanquantile(frac_infected_trials, 0.975, axis = 0)[1:(minmax_deg + 1)])
        lower_lst.append(np.nanquantile(frac_infected_trials, 0.025, axis = 0)[1:(minmax_deg + 1)])
        degree_lst.append(np.arange(1, minmax_deg + 1))
        frac_infected_all = np.mean(frac_infected_all_trials)
        frac_infected_all_lst.append(frac_infected_all)
        
    return frac_infected_lst, upper_lst, lower_lst, degree_lst, frac_infected_all_lst

# Get number of infected nodes per threshold
frac_infected_lst = []
upper_lst = []
lower_lst = []

batch_size = int(math.ceil(len(THRES_UP_LST) / n_cpu))
results = Parallel(n_jobs = n_cpu)(delayed(thres_trials_single)(
    THRES_UP_LST[(i * batch_size):min((i + 1) * batch_size, len(THRES_UP_LST))]
) for i in range(n_cpu))

for res in results:
    frac_single, upper_single, lower_single = res
    frac_infected_lst += frac_single
    upper_lst += upper_single
    lower_lst += lower_single
plt.plot(THRES_UP_LST, frac_infected_lst)
plt.fill_between(THRES_UP_LST, lower_lst, upper_lst, alpha = 0.1)
plt.axvline(x = CUTOFF, color = "red")
# plt.axvline(x = CUTOFF / 2, color = "green")
plt.axhline(y = 1, color = "black")
plt.xlabel("Maximum Threshold")
plt.ylabel("Fraction of Infected")
# plt.title(f"rho = {FRAC_INFECTED}, p = {P}")
plt.savefig(f"thres_{GRAPH_NAME}.png")
plt.clf()
plt.close()

# # Get number of infected nodes per degree per threshold
# frac_infected_lst = []
# upper_lst = []
# lower_lst = []

# batch_size = int(math.ceil(len(THRES_UP_LST) / n_cpu))
# results = Parallel(n_jobs = n_cpu)(delayed(degree_trials_single)(
#     THRES_UP_LST[(i * batch_size):min((i + 1) * batch_size, len(THRES_UP_LST))]
# ) for i in range(n_cpu))

# frac_infected_lst = []
# upper_lst = []
# lower_lst = []
# degree_lst = []
# frac_infected_all_lst = []

# for res in results:
#     frac_single, upper_single, lower_single, degree_single, frac_infected_all_trials = res
#     frac_infected_lst += frac_single
#     upper_lst += upper_single
#     lower_lst += lower_single
#     degree_lst += degree_single
#     frac_infected_all_lst += frac_infected_all_trials

# for i in range(len(THRES_UP_LST)):
#     plt.plot(degree_lst[i], frac_infected_lst[i])
#     plt.fill_between(degree_lst[i], lower_lst[i], upper_lst[i], alpha = 0.1)
#     #    plt.axvline(x = 1, color = "red", label = "degree = 1")
#     plt.axhline(y = frac_infected_all_lst[i], color = "green", label = "Pop Avg Infected")
#     plt.xlabel("Node Degree")
#     plt.ylabel("Fraction of Infected")
#     plt.legend()
#     plt.ylim(0, 1.1)
# #     plt.title(f"1/m = {round(THRES_UP_LST[i], 2)}, Cutoff = {round(CUTOFF, 2)}\nrho = {FRAC_INFECTED}, p = {P}")
#     plt.savefig(f"degree_{GRAPH_NAME}_maxthres={round(THRES_UP_LST[i], 2)}.png")
#     plt.clf()
#     plt.close()

