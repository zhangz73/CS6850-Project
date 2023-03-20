import numpy as np
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
from bokeh.io import output_notebook, show
from ndlib.viz.bokeh.DiffusionTrend import DiffusionTrend

## Global Constants
N = 1000
P = 0.01
THRES_LST = np.random.uniform(0, 1, size = N) #0.005
FRAC_INFECTED = 0.05
NUM_ITR = 20

# Network topology
g = nx.erdos_renyi_graph(N, P)

# Model selection
model = ep.ThresholdModel(g)

# Model Configuration
config = mc.Configuration()
config.add_model_parameter("fraction_infected", FRAC_INFECTED)

# Setting node parameters
for i in g.nodes():
    config.add_node_configuration("threshold", i, THRES_LST[i])

model.set_initial_status(config)

# Simulation execution
iterations = model.iteration_bunch(NUM_ITR)
trends = model.build_trends(iterations)
print(trends)
