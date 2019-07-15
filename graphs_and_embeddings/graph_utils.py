import networkx as nx
from utils import *
from s3 import S3Manager

class GraphBuilder(object):

	def __init__(self):
		self.s3_client = S3Manager(bucket_name='meetinggraphs')

	def buildKPGraph(self, candidate_phrases,dir_graph=True):

		"""
        Build a graph with key-phrases as nodes
        Args:
            candidate_phrases : Candidate phrases typically noun phrases and any other task specific POS sequences
            dir_graph: undirected/directed graph. Builds directed graph by default

        Returns:
            graph with nodes as candidate key-phrases without edge weights
        """

		if dir_graph==True:
			kpGraph = nx.DiGraph()
		else:
			kpGraph = nx.Graph()

		kpGraph.add_nodes_from(candidate_phrases)

		return kpGraph

	def addNodes(self, graphObj, node_list):

		graphObj.add_nodes_from(node_list)

		return graphObj

	def addNodeProperties(self, graphObj, dict_,propetyName=None):

		nx.set_node_attributes(graphObj, values = dict_, name=propetyName)

		return graphObj

	def formEdgesWithNodeSimilarity(self, graphObject, nodeFeatureDict, nodeWeightDict,edge_retention_perc=0.1, removeDanglingNodes=True):

		"""
        Does following:
            - Connects nodes with edges weight as cosine similarity between the associated feature vectors
            - If the graph is directional, the edge formed is node_with_lower_nodeweight -> node_with_higher_nodeweight. Otherwise, edges are connected in both directions
            - retains only top `(edge_retention_perc*100)%`	 edges and drops all others

        Args:
            graphObject : A networkx graph with candidate key-phrases as nodes
            nodeFeatureDict : A pre-built {keyphrase: embedding} dictionary with dictionary entry for each graph node
            nodeWeightDict : A pre-built {keyphrase: weight} dictionary with dictionary entry for each graph node - relevance measure between candidate keyphrase and the sentence it is part of
			edge_retention_perc : number of edges to be retained out of n*(n-1) edges formed - defaults to top 10% of the edge weights
			removeDanglingNodes : pruning edges from the key-phrase graph can result in dangling nodes. If removeDanglingNodes is set to `True`, such nodes will be removed
        Returns:
            graph with nodes as candidate key-phrases without edge weights
        """

		graphNodes = list(graphObject.nodes())
		edge_weight_list = []

		for i in range(len(graphNodes)):
			curr_node = graphNodes[i]
			if curr_node in graphObject.nodes():
				for j in range(i+1,len(graphNodes)):
					nxt_node = graphNodes[j]
					if nxt_node in graphObject.nodes():
						if validVectorCheck(nodeFeatureDict[curr_node]) and validVectorCheck(nodeFeatureDict[nxt_node]):
							edge_weight = CosineSim(nodeFeatureDict[curr_node],nodeFeatureDict[nxt_node])
						else:
							edge_weight = 0.001
						edge_weight_list.append(edge_weight)
						if nx.is_directed(graphObject):
							node1,node2 = self.getDirEdge(curr_node,nxt_node,nodeWeightDict)
							if node1!=node2:
								graphObject.add_edge(node1,node2,weight=edge_weight)
						else:
							graphObject.add_edge(node1,node2,weight=edge_weight)
							graphObject.add_edge(node2,node1,weight=edge_weight)

		edge_weight_list.sort(reverse=True)
		edge_weight_list = edge_weight_list[0:int(len(edge_weight_list)*edge_retention_perc)]
		min_edge_dist = min(edge_weight_list)
		drop_list = []
		edge_list = graphObject.edges

		drop_list = [edge for edge in edge_list if graphObject[edge[0]][edge[1]]['weight']<min_edge_dist]
				
		graphObject.remove_edges_from(drop_list)
		
		if removeDanglingNodes:
			graphObject.remove_nodes_from(list(nx.isolates(graphObject)))

		return graphObject

	def getDirEdge(self, curr_node,nxt_node,personalization_dict):

		curr_node_score = personalization_dict[curr_node]
		nxt_node_score = personalization_dict[nxt_node]
		if curr_node_score>nxt_node_score:
			order_ = nxt_node,curr_node
		else:
			order_ = curr_node,nxt_node
		return order_

	def getPageRankNodeWeights(self, graphObject, alpha=0.85, max_iter=100,tol=0.0001, personalization=None, nstart=None):
		node_weights = nx.pagerank(graphObject, alpha=alpha, max_iter=max_iter,tol=tol, personalization=personalization, nstart=nstart)
		return node_weights

	def uploadGraphObject(self, graph_object, file_name):
		#create pickle buffer and upload to the s3 bucket
		graph_pickle_dumps = pickleObject(graph_object)
		upload_resp = self.s3_client.upload_object(graph_pickle_dumps,file_name)

	def downloadGraphObject(self, s3Path):
		graphObject = self.s3_client.download_file(s3Path)
		return graphObject

