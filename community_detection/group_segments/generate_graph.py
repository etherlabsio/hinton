import networkx as nx
import json
import numpy as np
from scipy.spatial.distance import cosine
import statistics

import pickle



# marketing
#ent_fv_full = pickle.load(open("/home/arjun/NER_experiments/code/entity_graph_builder/graph_dumps/marketing_entity_feats_marketing_model_epc3.pkl","rb"))
#ent_graph = pickle.load(open("/home/arjun/NER_experiments/code/entity_graph_builder/graph_dumps/entity_kp_graph_marketing.pkl","rb"))

# S.E
ent_fv_full = pickle.load(open("/home/arjun/NER_experiments/code/entity_graph_builder/graph_dumps/se_entity_feats_se_model_v2epc3.pkl", "rb"))
#ent_graph = pickle.load(open("/home/ether/hdd/Venkat/knowledge_graphs/entity_graph_builder/graph_dumps/pruned_entity_kp_graph.pkl","rb"))
ent_graph = pickle.load(open("/home/arjun/NER_experiments/code/entity_graph_builder/graph_dumps/entity_kp_graph_directed_sev2_with_synrel.pkl", "rb"))
# Ether Graph
# ent_graph = pickle.load(open("/home/ether/hdd/Venkat/knowledge_graphs/entity_graph_builder/graph_dumps/se_ether_graph_slack_extended.pkl","rb"))
# ent_fv_full = pickle.load(open("/home/ether/hdd/Venkat/knowledge_graphs/entity_graph_builder/graph_dumps/ether_engg_entity_feats_+slack_ether_model_2+1_epc3.pkl","rb"))



common_entities = ent_fv_full.keys() & ent_graph.nodes()
ent_fv = {}
for ent in common_entities:
    ent_fv[ent] = ent_fv_full[ent]
    

G = nx.Graph()

G.add_nodes_from(list(ent_fv.keys()))

for nodea in G.nodes():
    for nodeb in G.nodes():
        G.add_edge(nodea, nodeb, weight= 1-cosine(ent_fv[nodea], ent_fv[nodeb]))
        

with open("new_graph", "wb") as f:
    pickle.dump([G.nodes(data=True), G.edges(data=True)], f)