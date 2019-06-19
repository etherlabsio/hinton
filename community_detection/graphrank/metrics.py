# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import networkx as nx
import logging

logger = logging.getLogger(__name__)


class GraphSolvers(object):
    """
    Class for computing various graph properties
    """
    def __init__(self):
        pass

    def get_graph_algorithm(self, graph_obj, solver_fn,personalization=None):
        """
        Graph algorithms to compute nodes/vertices' weights
        Args:
            graph_obj (networkx graph obj): Graph object whose nodes need to be weighted
            solver_fn (str): Name of the graph algorithm to use for calculating weights

        Returns:
            node_weights (list[tuple]): List of tuple(Nodes, weighted scores)
        """

        # TODO Extend to other solvers
        if solver_fn == 'pagerank':
            node_weights = nx.pagerank(graph_obj, alpha=0.85, tol=0.0001, weight=None, personalization=personalization)
        else:
            node_weights = nx.pagerank_scipy(graph_obj, alpha=0.85, tol=0.0001, weight=None)

        return node_weights

    def get_node_degree(self, graph_obj):
        node_degrees = nx.degree(graph_obj)

        return dict(node_degrees)

    def get_betweenness(self, graph_obj):
        if nx.is_connected(graph_obj):
            node_betweenness = nx.current_flow_betweenness_centrality(graph_obj)
        else:
            node_betweenness = nx.betweenness_centrality(graph_obj)

        return dict(node_betweenness)

    def get_closeness(self, graph_obj):

        if nx.is_connected(graph_obj):
            node_closeness = nx.current_flow_closeness_centrality(graph_obj)
        else:
            node_closeness = nx.closeness_centrality(graph_obj)

        return dict(node_closeness)

    def normalize_nodes(self, graph_obj, node_weights, normalize_fn=None):
        """
        Normalize node weights using graph properties
        Args:
            graph_obj:
            node_weights:
            normalize_fn:

        Returns:

        """
        if normalize_fn == 'degree':
            node_degrees = self.get_node_degree(graph_obj=graph_obj)
            for k, v in node_weights.items():
                try:
                    node_weights[k] = v/node_degrees[k]
                except Exception as e:
                    logger.debug("Zero degree value while computing degree: {}".format(e))
                    node_weights[k] = v

        elif normalize_fn == 'closeness':
            node_closeness = self.get_closeness(graph_obj=graph_obj)
            for k, v in node_weights.items():
                try:
                    node_weights[k] = v/node_closeness[k]
                except Exception as e:
                    logger.debug("Zero degree value while computing closeness: {}".format(e))
                    node_weights[k] = v

        elif normalize_fn == 'betweenness':
            node_betweenness = self.get_betweenness(graph_obj=graph_obj)
            for k, v in node_weights.items():
                try:
                    node_weights[k] = v * node_betweenness[k]
                except Exception as e:
                    logger.debug("Zero value while computing betweenness: {}".format(e))
                    node_weights[k] = v

        elif normalize_fn == 'degree_bet':
            node_degrees = self.get_node_degree(graph_obj)
            node_bet = self.get_betweenness(graph_obj)
            for k, v in node_weights.items():
                node_norm = node_degrees[k] + node_bet[k]
                node_weights[k] = v/node_norm
        else:
            node_weights = node_weights

        return node_weights


class WeightMetrics(object):
    """
    Class for computing aggregated weight scores for multiple nodes. The aggregation funtion helps in forming and
    ranking keyphrases.
    """
    def compute_weight_fn(self, weight_metrics, key_terms, score_list, normalize=False):
        weighted_score = 0

        if weight_metrics == 'max':
            weighted_score = self.compute_max_score(key_terms=key_terms, score_list=score_list, normalize=normalize)
        elif weight_metrics == 'sum':
            weighted_score = self.compute_sum_score(key_terms=key_terms, score_list=score_list, normalize=normalize)
        else:
            weighted_score = self.compute_max_score(key_terms=key_terms, score_list=score_list, normalize=normalize)

        # TODO extend to more weighting metrics

        return weighted_score

    def compute_max_score(self, key_terms, score_list, normalize=False, threshold=3):
        unit_size = len(key_terms)
        if unit_size > threshold and normalize:
            weight_score = max(score_list)/float(unit_size)
        else:
            weight_score = max(score_list)

        return weight_score

    def compute_sum_score(self, key_terms, score_list, normalize=False, threshold=3):
        unit_size = len(key_terms)
        if unit_size > threshold and normalize:
            weight_score = sum(score_list)/float(unit_size)
        else:
            weight_score = sum(score_list)

        return weight_score

