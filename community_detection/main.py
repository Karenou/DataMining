import pandas as pd
import networkx as nx
from networkx.algorithms import community
from sklearn.metrics.cluster import normalized_mutual_info_score

# read data and construct graph
G = nx.read_edgelist("community_detection/dataset/email-Eu-core.txt", create_using=nx.DiGraph)
labels = pd.read_csv("community_detection/dataset/email-Eu-core-department-labels.txt", header=None, delimiter=' ')

# construct the weak undirected graph
Gw = max(nx.weakly_connected_component_subgraphs(G), key=len)
G_weak = G.subgraph(Gw)
G_w = G_weak.to_undirected()

# number of communities to be found
k = 42
comp = community.asyn_fluid.asyn_fluidc(G_w, k, max_iter=500, seed=100)

# number of nodes in the graph
N = 1005
pred_dict = {"node id": [], "department id": []}
pred = list(range(N))
for index, comm in enumerate(comp):
    for p in comm:
        pred[int(p)] = index
        pred_dict["node id"].append(int(p))
        pred_dict["department id"].append(index)

print("The normalized mutual information is %.4f" % normalized_mutual_info_score(labels[1].values, pred))

# save output
pred_df = pd.DataFrame.from_dict(pred_dict, orient="columns")
pred_df.to_csv("community_detection/result.csv", index=False, header=True)
