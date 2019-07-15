import networkx as nx
from graph_utils import GraphBuilder
import pickle

graphBuilder = GraphBuilder()
s3_graph_object = graphBuilder.downloadGraphObject('samplegraph')
file_obj_bytestring = s3_graph_object["Body"].read()
graph_obj = graph_obj = pickle.loads(file_obj_bytestring)

print(graph_obj.nodes())