import numpy as np
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from bokeh.io import output_notebook, show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend
import matplotlib.pyplot as plt
from tqdm import tqdm

## Global Constants
N = 100 #1000
P_LST = np.linspace(0, 0.5, 101)[1:]
THRES_LST = np.random.uniform(0, 1, size = N) #0.005
FRAC_INFECTED = 0.25
NUM_ITR = 20
NUM_TRIALS = 20

# Model Configuration
config = mc.Configuration()
config.add_model_parameter("fraction_infected", FRAC_INFECTED)
# Setting node parameters
for i in range(N):
    config.add_node_configuration("threshold", i, THRES_LST[i])

def single_trial(P, get_graph = False):
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

P = 0.1
frac_infected, (g, snapshot) = single_trial(P, get_graph = True)
color_map = []
for node in g:
    if snapshot[node] == 1:
        color_map.append("#ffc3d7")
    else:
        color_map.append("#badaff")
cent=nx.edge_betweenness_centrality(g)
node_pos=nx.spring_layout(g)
nx.draw_networkx_nodes(g, pos = node_pos, node_color = color_map, node_size = 50)
nx.draw_networkx_edges(g, pos = node_pos, edgelist = cent, alpha = 0.1)
plt.savefig(f"Plots/snapshot_p={P}_rho={FRAC_INFECTED}.png")
plt.clf()
plt.close()

#frac_infected_lst = []
#upper_lst = []
#lower_lst = []
#for P in tqdm(P_LST):
#    frac_infected_trials = np.zeros(NUM_TRIALS)
#    for trial in range(NUM_TRIALS):
#        frac_infected, _ = single_trial(P)
#        frac_infected_trials[trial] = frac_infected
#    frac_infected_lst.append(np.mean(frac_infected_trials))
#    upper_lst.append(np.quantile(frac_infected_trials, 0.975))
#    lower_lst.append(np.quantile(frac_infected_trials, 0.025))
#
#plt.plot(P_LST, frac_infected_lst)
#plt.fill_between(P_LST, lower_lst, upper_lst, alpha = 0.1)
#plt.xlabel("P")
#plt.ylabel("Fraction of Infected")
#plt.title(f"rho = {FRAC_INFECTED}")
#plt.savefig(f"Plots/linearThres_rho={FRAC_INFECTED}.png")
#plt.clf()
#plt.close()
