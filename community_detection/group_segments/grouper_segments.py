# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: sri_gpt
#     language: python3
#     name: sri_gpt
# ---

# +


#from scipy.spatial.distance import cityblock
import sys
sys.path.append("/home/ray__/ssd/BERT/")
from gpt_feat_utils import GPT_Inference
sys.path.append("../../../ai-engine_temp/pkg/")
import copy
import numpy as np
import json
from copy import deepcopy
import text_preprocessing.preprocess as tp
import extra_preprocess
# from group_segments.extra_preprocess import format_time
import networkx as nx
import math
from scorer import cosine
import community
from datetime import datetime
from scorer import get_feature_vector, get_embeddings, get_feature_vector_local
from collections import Counter
import logging
import numpy as np
from log.logger import setup_server_logger
from chinese_whispers import chinese_whispers, aggregate_clusters
logger = logging.getLogger()



#
# #gpt_model = gpt_feat_utils.GPT_SimInference("/home/arjun/gpt_experiments/models/model_lm+sim_ep3/", device="cuda")
# #gpt_model = gpt_feat_utils.GPT_SimInference("/home/arjun/gpt_experiments/models/model_lm+nsp_sim_ep3/", device="cuda")
#gpt_model = GPT_Inference("/home/arjun/gpt_experiments/engg_models/se+ether_2+1s_ep5_#2/", device="cpu")
#gpt_model = GPT_Inference("/home/ray__/ssd/BERT/models/product/", device="cuda")
#gpt_model = GPT_Inference("/home/ray__/ssd/BERT/models/ether_v2/ether_googleJan13_groupsplit_withstop_4+w_gt3s_lr3e-5/",device="cpu")
gpt_model = GPT_Inference("/home/ray__/ssd/BERT/models/customer_service/epoch3/", device="cpu")

# -


class community_detection():
    segments_list = []
    segments_org = []
    segments_map = {}
    segments_order = {}
    lambda_function = None
    mind_features = None
    mind_id = None
    context_id = None
    instance_id = None
    compute_fv = True

    def __init__(self, Request, lambda_function,compute_fv=True):
        self.segments_list = Request.segments
        self.segments_org = Request.segments_org
        self.segments_order = Request.segments_order
        self.segments_map = Request.segments_map
        self.lambda_function = lambda_function
        self.compute_fv = compute_fv
        print ("Using ", self.mind_id, " for feature extraction")
        self.context_id = Request.context_id
        self.instance_id = Request.instance_id

    # def compute_feature_vector(self):
    #     graph_list = {}
    #     fv = {}
    #     index = 0
    #     for segment in self.segments_list:
    #         for sent in segment['originalText']:
    #             if sent!='':
    #                 graph_list[index] = (sent, segment['startTime'], segment['spokenBy'], segment['id'])
    #                 fv[index] = getBERTFeatures(self.model1, sent, attn_head_idx=-1)
    #                 index+=1
    #     return fv, graph_list

    def compute_feature_vector(self):
        graph_list = {}
        fv = {}
        index = 0
        all_segments = ""
        for segment in self.segments_list:
            for sent in segment['originalText']:
                if sent != '':
                    if sent[-1] == ".":
                        all_segments = all_segments + " " + sent
                    else:
                        all_segments = all_segments + " " + sent + ". "
        mind_input = json.dumps({"text": all_segments, "nsp": False})
        mind_input = json.dumps({"body": mind_input})
        transcript_score = get_feature_vector(mind_input, self.lambda_function)
        for segment in self.segments_list:
            for sent in segment['originalText']:
                if sent != '':
                    graph_list[index] = (sent, segment['startTime'], segment['spokenBy'], segment['id'])
                    fv[index] = transcript_score[index]
                    index += 1
        return fv, graph_list
    
    def compute_feature_vector_gpt(self):
        graph_list = {}
        input_list = []
        fv = {}
        index = 0
        for segment in self.segments_list:
            for sent in segment['originalText']:
                if sent != '':
                    input_list.append(sent)
        transcript_score = get_feature_vector(input_list, self.lambda_function)
        for segment in self.segments_list:
            for sent in segment['originalText']:
                if sent != '':
                    graph_list[index] = (sent, segment['startTime'], segment['spokenBy'], segment['id'])
                    fv[index] = transcript_score[index]
                    index += 1
        assert(len(graph_list) == len(fv))
        return fv, graph_list


    def compute_feature_vector_gpt(self, local=False):
        graph_list = {}
        fv_mapped_score = {}
        input_list = []
        fv = {}
        index = 0
        for segment in self.segments_list:
            for sent in segment['originalText']:
                if sent != '':
                    input_list.append(sent)
        if not local:
            transcript_score, mind_score = get_feature_vector(input_list, self.lambda_function, self.mind_features)
        else:
            transcript_score = get_feature_vector_local(input_list, self.lambda_function, gpt_model)
        for segment in self.segments_list:
            for sent in segment['originalText']:
                if sent != '':
                    graph_list[index] = (sent, segment['startTime'], segment['spokenBy'], segment['id'])
                    fv[index] = transcript_score[index]
                 
                    # fv_mapped_score[index] = (segment['id'], mind_score[index])
                    index += 1
        return fv, graph_list
    
    def compute_feature_vector_gpt_para(self):
        graph_list = {}
        input_list = []
        fv = {}
        index = 0
        for segment in self.segments_list:
            input_list = []
            for sent in segment["originalText"]:
                if sent != '':
                    input_list.append(sent)
            if input_list!=[]:        
                transcript_score = get_feature_vector(input_list, self.lambda_function)
                fv[index] = np.mean(transcript_score, axis=0)
                graph_list[index] = (" ".join(segment["originalText"]), segment['startTime'], segment['spokenBy'], segment['id'])
                index +=1
        return fv, graph_list
    
    def compute_feature_vector_use(self):
        graph_list = {}
        input_list = []
        fv = {}
        index = 0
        for segment in self.segments_list:
            for sent in segment['originalText']:
                if sent != '':
                    input_list.append(sent)
                    
        transcript_score = get_embeddings(input_list)
        for segment in self.segments_list:
            for sent in segment['originalText']:
                if sent != '':
                    graph_list[index] = (sent, segment['startTime'], segment['spokenBy'], segment['id'])
                    fv[index] = transcript_score[index]
                    index += 1
        return fv, graph_list

    def construct_graph_vanilla(self, fv, graph_list):
        meeting_graph = nx.Graph()
        yetto_prune = []
        c_weight = 0
        for nodea in graph_list.keys():
            for nodeb in graph_list.keys():
                c_weight = cosine(fv[nodea], fv[nodeb])
                meeting_graph.add_edge(nodea, nodeb, weight=c_weight)
                yetto_prune.append((nodea, nodeb, c_weight))
        return meeting_graph, yetto_prune

    def construct_graph(self, fv, graph_list):
        meeting_graph = nx.Graph()
        yetto_prune = []
        c_weight = 0
        for nodea in graph_list.keys():
            for nodeb in graph_list.keys():
                #if nodeb > nodea:
                c_weight = cosine(fv[nodea], fv[nodeb])
                # c_weight =  -gpt_model.get_feat_sim(fv[nodea], fv[nodeb])
                #c_weight = 1 - cityblock(fv[nodea], fv[nodeb])
                # print (nodea, nodeb)
                #if self.segments_order[graph_list[nodeb][-1]] == 1 + self.segments_order[graph_list[nodea][-1]]:
                #c_weight +=  -gpt_model.get_feat_sim(fv[nodea], fv[nodeb])
                meeting_graph.add_edge(nodea, nodeb, weight=c_weight)
                yetto_prune.append((nodea, nodeb, c_weight))
                
#                 if self.segments_order[graph_list[nodea][-1]] - self.segments_order[graph_list[nodeb][-1]] == (0 or -1):
#                     c_weight = cosine(fv[nodea], fv[nodeb])
#                     meeting_graph.add_edge(nodea, nodeb, weight=c_weight)
#                     yetto_prune.append((nodea, nodeb, c_weight))

        logger.info("Normalising the Graph", extra={"nodes: ":meeting_graph.number_of_nodes(), "edges: ": meeting_graph.number_of_edges()})
        X = nx.to_numpy_array(meeting_graph)

        for i in range(len(X)):
            X[i][i] = X[i].mean()

        norm_mat = (X - X.min(axis=1)) / (X.max(axis=1) - X.min(axis=1))
        norm_mat = (np.transpose(np.tril(norm_mat)) + np.triu(norm_mat)) / 2
        norm_mat = norm_mat + np.transpose(norm_mat)

        meeting_graph = nx.from_numpy_array(norm_mat)
        logger.info("Completed Normalization", extra={"nodes: ":meeting_graph.number_of_nodes(), "edges: ": meeting_graph.number_of_edges()})
        #meeting_graph.remove_edges_from(list(map(lambda x: (x,x), range(meeting_graph.number_of_nodes()))))
        
        for index in range(meeting_graph.number_of_nodes()):
            meeting_graph[index][index]["weight"] = 1
    
        logger.info("Completed Normalization and after removing diagonal values", extra={"nodes: ":meeting_graph.number_of_nodes(), "edges: ": meeting_graph.number_of_edges()})
        for i,(nodea,nodeb,score) in enumerate(yetto_prune):
            yetto_prune[i] = (nodea, nodeb, meeting_graph[nodea][nodeb]["weight"])
            
#         for node in list(meeting_graph.nodes()):
#             meeting_graph.nodes[node]["text"] = graph_list[node][0]
#             meeting_graph.nodes[node]["tag"] = self.segments_map[graph_list[node][-1]]["tag"]
        return meeting_graph, yetto_prune
    
    def construct_graph_ns(self, fv, graph_list):
        meeting_graph = nx.Graph()
        yetto_prune = []
        c_weight = 0
        for nodea in graph_list.keys():
            for nodeb in graph_list.keys():
                if ((self.segments_order[graph_list[nodeb][-1]] - self.segments_order[graph_list[nodea][-1]]) in [0, 1]):
                    #c_weight +=  -gpt_model.get_feat_sim(fv[nodea], fv[nodeb])
                    c_weight = cosine(fv[nodea], fv[nodeb])
                    #c_weight = np.linalg.norm(np.asarray(fv[nodea]) - np.asarray(fv[nodeb]))
                    meeting_graph.add_edge(nodea, nodeb, weight=c_weight)
                    yetto_prune.append((nodea, nodeb, c_weight))

#         logger.info("Normalising the Graph", extra={"nodes: ":meeting_graph.number_of_nodes(), "edges: ": meeting_graph.number_of_edges()})
        
#         X = nx.to_numpy_array(meeting_graph)

#         for i in range(len(X)):
#             X[i][i] = X[i].mean()

#         norm_mat = (X - X.min(axis=1)) / (X.max(axis=1) - X.min(axis=1))
#         norm_mat = (np.transpose(np.tril(norm_mat)) + np.triu(norm_mat)) / 2
#         norm_mat = norm_mat + np.transpose(norm_mat)

#         meeting_graph = nx.from_numpy_array(norm_mat)
#         logger.info("Completed Normalization", extra={"nodes: ":meeting_graph.number_of_nodes(), "edges: ": meeting_graph.number_of_edges()})
        
#         for index in range(meeting_graph.number_of_nodes()):
#             meeting_graph[index][index]["weight"] = 1
    
#         logger.info("Completed Normalization and after removing diagonal values", extra={"nodes: ":meeting_graph.number_of_nodes(), "edges: ": meeting_graph.number_of_edges()})
# #         for i,(nodea,nodeb,score) in enumerate(yetto_prune):
# #             yetto_prune[i] = (nodea, nodeb, meeting_graph[nodea][nodeb]["weight"])
        
#         yetto_prune = []
#         for nodea, nodeb, weight in meeting_graph.edges.data():
#             yetto_prune.append((nodea, nodeb, weight["weight"]))
        return meeting_graph, yetto_prune

    def construct_graph_ns_max(self, fv, graph_list):
        meeting_graph = nx.Graph()
        yetto_prune = []
        c_weight = 0

        for nodea in graph_list.keys():
            for nodeb in graph_list.keys():
                c_weight = cosine(fv[nodea], fv[nodeb])
                meeting_graph.add_edge(nodea, nodeb, weight=c_weight)
                yetto_prune.append((nodea, nodeb, c_weight))
        import statistics
        max_connection = {}
        max_score = {}
        lowest_score = {}
        #avg_score = {}
        outlier_score = {}
        for node in meeting_graph.nodes():
            closest_connection_n = sorted(dict(meeting_graph[node]).items(), key=lambda kv:kv[1]["weight"], reverse=True)
            closest_connection = []
            max_score_current = []
            lowest_score_current = []
            for n, score in closest_connection_n:
                if ((self.segments_order[graph_list[node][-1]] <= self.segments_order[graph_list[n][-1]]) and (self.segments_order[graph_list[n][-1]] - self.segments_order[graph_list[node][-1]] in [0, 1])):
                    max_score_current.append((n ,score))
                else:
                    closest_connection.append((n, score))
                lowest_score_current.append((n, score))
            #avg_score[node] = np.mean(list(map(lambda kv: kv[1]["weight"], max_score_current)))
            #max_score[node] = sorted(max_score_current, key=lambda kv:kv[1]["weight"], reverse=False)[1]
            lowest_score[node] = sorted(lowest_score_current, key=lambda kv:kv[1]["weight"], reverse=False)[:10]
            #max_connection[node] = [(n1, score1["weight"]) for (n1, score1) in closest_connection if score1["weight"] > max_score[node][1]["weight"]]
            max_connection[node] = closest_connection_n
            weights_n = list(map(lambda kv: (kv[1]["weight"]).tolist(), closest_connection_n))
            q3 = np.percentile(weights_n, 75)
            iqr = np.subtract(*np.percentile(weights_n, [75, 25]))
            outlier_score[node] = {}
            outlier_score[node]["outlier"] = q3 + 1 * iqr
            outlier_score[node]["iqr"] = iqr
            outlier_score[node]["q3"] = q3
            outlier_score[node]["weights_n"] = closest_connection_n
            outlier_score[node]["avg+pstd"] = statistics.mean(weights_n)+statistics.pstdev(weights_n)


        graph_data = deepcopy(meeting_graph.edges.data())
        for nodea, nodeb, weight in graph_data:
            #if nodeb in list(map(lambda kv: kv[0], max_connection[nodea])) or nodeb in list(map(lambda kv: kv[0], lowest_score[nodea])) or ((self.segments_order[graph_list[nodeb][-1]] - self.segments_order[graph_list[nodea][-1]]) in [0, 1]):
            #if nodeb in list(map(lambda kv: kv[0], max_connection[nodea])) or nodeb in list(map(lambda kv: kv[0], lowest_score[nodea])):
            #if weight["weight"]>avg_score[nodea]:
            #if nodeb in list(map(lambda kv:kv[0], max_connection[nodea])):
            #if nodeb in list(map(lambda kv:kv[0], max_connection[nodea])) or (((self.segments_order[graph_list[nodeb][-1]] - self.segments_order[graph_list[nodea][-1]]) in [0])):
            if weight["weight"] > outlier_score[nodea]["outlier"] or (((self.segments_order[graph_list[nodeb][-1]] - self.segments_order[graph_list[nodea][-1]]) in [0])):
                #print (True)
                #    pass
                #else:
                #    print (graph_list[nodea][0], " ==> ", graph_list[nodeb][0])
                #if nodeb in list(map(lambda kv: kv[0], lowest_score[nodea])):
                #    print (graph_list[nodea][0], "!==", graph_list[nodeb][0])
                pass
            elif (self.segments_order[graph_list[nodeb][-1]] - self.segments_order[graph_list[nodea][-1]]) in [-2, -1, 1, 2] and weight["weight"] > outlier_score[nodea]["avg+pstd"] :
                    print (graph_list[nodea], graph_list[nodeb])
                    pass
            else:
                meeting_graph.remove_edge(nodea, nodeb)
        for nodea, nodeb, weight in graph_data:
            if (self.segments_order[graph_list[nodeb][-1]] - self.segments_order[graph_list[nodea][-1]]) in [0]:
                #meeting_graph[nodea][nodeb]["Weight"] = outlier_score[nodea]["weights_n"][1][1]["weight"]
                meeting_graph[nodea][nodeb]["weight"] = 1
#         X = nx.to_numpy_array(meeting_graph)

#         for i in range(len(X)):
#             X[i][i] = X[i].mean()

#         norm_mat = (X - X.min(axis=1)) / (X.max(axis=1) - X.min(axis=1))
#         norm_mat = (np.transpose(np.tril(norm_mat)) + np.triu(norm_mat)) / 2
#         norm_mat = norm_mat + np.transpose(norm_mat)

#         meeting_graph = nx.from_numpy_array(norm_mat)

#         for index in range(meeting_graph.number_of_nodes()):
#             meeting_graph[index][index]["weight"] = 1


        #import pickle
        #with open("meeting_graph", "wb") as f:
        #    pickle.dump([meeting_graph.nodes(data=True), meeting_graph.edges(data=True), graph_list], f)
        return meeting_graph, yetto_prune

    def prune_edges_outlier(self, meeting_graph, graph_list, yetto_prune, v):
        meeting_graph_pruned = nx.Graph()
        weights = []
        for nodea, nodeb, weight in meeting_graph.edges.data():
            meeting_graph_pruned.add_nodes_from([nodea, nodeb])
            weights.append(weight["weight"])

        # q1 = np.percentile(weights, 25)
        # iqr = np.subtract(*np.percentile(weights, [75, 25]))
        # outlier = q1 - 1.5 * iqr
        q3 = np.percentile(weights, v)
        # logger.info("Outlier Score", extra={"outlier threshold is : ": outlier})
        logger.info("Outlier Score", extra={"outlier threshold is : ": q3})
        removed = 0
        total = 0
        for indexa, indexb, c_score in meeting_graph.edges.data():
            total+=1
            if c_score["weight"]>=q3:
                removed+=1
                meeting_graph_pruned.add_edge(indexa, indexb, weight=c_score["weight"])
        # print(total-removed, removed)
        return meeting_graph_pruned

    def prune_edges(self, meeting_graph, graph_list, yetto_prune, v):
        yetto_prune = sorted(yetto_prune, key=lambda kv : kv[2], reverse=True)
        meeting_graph_pruned = nx.Graph()
        for nodea, nodeb, weight in yetto_prune:
            meeting_graph_pruned.add_nodes_from([nodea, nodeb])
        yetto_prune = yetto_prune[:math.ceil(len(yetto_prune) * v) + 1]
        logger.info("pruning value", extra={"v is : ": v})
        for indexa, indexb, c_score in yetto_prune:
            meeting_graph_pruned.add_edge(indexa, indexb)
        return meeting_graph_pruned

    def compute_louvian_community(self, meeting_graph_pruned, community_set):
        # community_set = community.best_partition(meeting_graph_pruned)
        # modularity_score = community.modularity(community_set, meeting_graph_pruned)
        # logger.info("Community results", extra={"modularity score":modularity_score})
        community_set_sorted = sorted(community_set.items(), key=lambda kv: kv[1], reverse=False)
        return community_set_sorted

    def refine_community(self, community_set_sorted, graph_list):
        clusters = []
        temp = []
        prev_com = 0
        seg_cls = {}
        seg_max = {}
        for index, (word, cluster) in enumerate(community_set_sorted):
            if cluster not in seg_cls.keys():
                seg_cls[cluster] = {}
            if prev_com == cluster:
                temp.append((word, graph_list[word][-1]))
                if index == len(community_set_sorted) - 1:
                    clusters.append(temp)
            else:
                clusters.append(temp)
                temp = []
                prev_com = cluster
                temp.append((word, graph_list[word][-1]))
        
        for cluster in clusters:
            print ("cluster before alteration=========>")
            for sent in cluster:
                #print (graph_list[sent[0]][0], "--- ", self.segments_map[graph_list[sent[0]][-1]]["tag"] ," ---")
                print (graph_list[sent[0]][0])
        # print (clusters)
        for index, cluster in enumerate(clusters):
            seg_cls[index] = Counter( seg for sent, seg in cluster)
            seg_count = {}
            for segid, count in seg_cls[index].items():
                seg_count[segid] = count
            for segid in seg_count.keys():
                if segid not in seg_max.keys():
                    seg_max[segid] = (seg_count[segid], index)
                elif seg_count[segid] >= seg_max[segid][0]:
                    seg_max[segid] = (seg_count[segid], index)
        # print (seg_max)
        
        print ("\n\n\n")
        new_clusters = deepcopy(clusters)
        for index, cluster in enumerate(new_clusters):
            for sent, seg in cluster:
                if seg_max[seg][1] != index:
                    clusters[index].remove((sent, seg))
                    
        # print (clusters)
        for cluster in clusters:
            print ("\nAfter removing overlapping groups")
            print ("cluster =========>\n")
            for sent in cluster:
                print (graph_list[sent[0]][0])
        timerange = []
        temp = []
        for cluster in clusters:
            temp = []
            for sent, seg in cluster:
                # temp.append(graph_list[sent])
                # logger.info("segment values", extra={"segment":self.segments_list})
                temp.append(graph_list[sent])
            if len(temp) != 0:
                temp = list(set(temp))
                temp = sorted(temp, key=lambda kv: kv[1], reverse=False)
                timerange.append(temp)
        print ("\n\nChecking Timerange --------------")
        for cls in timerange:
            print ("--------------")
            for sent in cls:
                print (sent[0], sent[-1], "\n")
        print ("<---------------->")
        return timerange

    def remove_preprocessed_segments(self, graph_list):
        # segments_order_modified = {}
        graph_list_id = list(map(lambda x: x[-1], graph_list.values()))
        temp_segments_order = deepcopy(list(self.segments_order.items()))
        temp_segments_order = sorted(temp_segments_order, key= lambda kv: kv[1], reverse=False)
        sudo_index = 0
        for segid, index in temp_segments_order:
            if segid not in graph_list_id:
                del self.segments_order[segid]
            else:
                self.segments_order[segid] = sudo_index
                sudo_index+=1
                
        for seg in self.segments_order.keys():
            print (self.segments_map[seg])
        return True
    def group_community_by_time(self, timerange):
        # print (self.segments_order)
        timerange_detailed = []
        temp = []
        flag = False
        pims = {}
        index_pim = 0
        index_segment = 0

        for index, com in enumerate(timerange):
            temp = []
            flag = False

            if com[1:] == []:
                pims[index_pim] = {'segment0': [com[0][0], com[0][1], com[0][2], com[0][3]]}
                index_pim += 1
                continue

            for (index1, (sent1, time1, user1, id1)), (index2, (sent2, time2, user2, id2)) in zip(enumerate(com[0:]), enumerate(com[1:])):
                #if id1 = id2:
                if True:
                    # if ((extra_preprocess.format_time(time2, True) - extra_preprocess.format_time(time1, True)).seconds <= 120):
                    if ((self.segments_order[id2] - self.segments_order[id1]) in [0, 1]):
                        print ("order difference:", (self.segments_order[id2] - self.segments_order[id1]))
                        print ("Relevant sentence: ", sent1 , "   =====   ", sent2)
                        if (not flag):
                            pims[index_pim] = {'segment' + str(index_segment): [sent1, time1, user1, id1]}
                            index_segment += 1
                            temp.append((sent1, time1, user1, id1))
                        pims[index_pim]['segment' + str(index_segment)] = [sent2, time2, user2, id2]
                        index_segment += 1
                        temp.append((sent2, time2, user2, id2))
                        flag = True
                    else:
                        print ("Not Relevant sentence: ", sent1 , "   !=   ", sent2)
                        print ("order difference:", (self.segments_order[id2] - self.segments_order[id1]))
                        if flag is True:
                            index_pim += 1
                            index_segment = 0
                        elif flag is False and index2 == len(com) - 1:
                            pims[index_pim] = {'segment0' : [sent1, time1, user1, id1]}
                            index_pim += 1
                            temp.append((sent1, time1, user1, id1))
                            pims[index_pim] = {'segment0' : [sent2, time2, user2, id2]}
                            index_pim += 1
                            temp.append((sent2, time2, user2, id2))
                        else:
                            pims[index_pim] = {'segment0' : [sent1, time1, user1, id1]}
                            index_pim += 1
                            temp.append((sent1, time1, user1, id1))
                        flag = False
            if flag is True:
                index_pim += 1
                index_segment = 0
            timerange_detailed.append(temp)
        #print("=====> check here")
        #print (pims)
        return pims

    def wrap_community_by_time(self, pims):
        yet_to_combine = []
        need_to_remove = []
        inverse_dangling_pims = []
        for index1, i in enumerate(pims.keys()):
            for index2, j in enumerate(pims.keys()):
                if index1 != index2:
                    if (pims[i]['segment0'][1] >= pims[j]['segment0'][1] and pims[i]['segment0'][1] <= pims[j]['segment' + str(len(pims[j].values()) - 1)][1]) and (pims[i]['segment' + str(len(pims[i].values()) - 1)][1] >= pims[j]['segment0'][1] and pims[i]['segment' + str(len(pims[i].values()) - 1)][1] <= pims[j]['segment' + str(len(pims[j].values()) - 1)][1]) :
                        if (j, i) not in yet_to_combine and i not in need_to_remove and j not in need_to_remove:
                            yet_to_combine.append((i, j))
                            need_to_remove.append(i)
        for i, j in yet_to_combine:
            for k in pims[i]:
                if pims[i][k] not in pims[j].values():
                    pims[j]['segment' + str(len(pims[j].values()))] = pims[i][k]
        for i in need_to_remove:
            pims.pop(i)

        for index, p in enumerate(pims.keys()):
            for seg in pims[p].keys():
                pims[p][seg][0] = [' '.join(text for text in segment['originalText']) for segment in self.segments_list if segment['id'] == pims[p][seg][3]]
                inverse_dangling_pims.append(pims[p][seg][3])

        c_len = 0
        for segment in self.segments_list:
            if segment['id'] not in inverse_dangling_pims:
                while c_len in pims.keys():
                    c_len += 1
                pims[c_len] = {"segment0": [' '.join(text for text in segment['originalText']), segment['startTime'], segment['spokenBy'], segment['id']]}
        return pims

    def wrap_community_by_time_refined_d(self, pims):
        inverse_dangling_pims = []
        pims_keys = list(pims.keys())

        for i in pims_keys:
            for j in pims_keys:
                if i != j and i in pims.keys() and j in pims.keys():
                    if (pims[i]['segment0'][1] >= pims[j]['segment0'][1] and pims[i]['segment0'][1] <= pims[j]['segment' + str(len(pims[j].values()) - 1)][1]) and (pims[i]['segment' + str(len(pims[i].values()) - 1)][1] >= pims[j]['segment0'][1] and pims[i]['segment' + str(len(pims[i].values()) - 1)][1] <= pims[j]['segment' + str(len(pims[j].values()) - 1)][1]):
                        for seg in pims[i].values():
                            pims[j]['segment' + str(len(pims[j].values()))] = seg
                        del pims[i]

                        sorted_j = sorted(pims[j].values(), key=lambda kv: kv[1], reverse=False)
                        temp_pims = {}
                        new_index = 0
                        for new_seg in sorted_j:
                            temp_pims['segment' + str(new_index)] = new_seg
                            new_index += 1
                        pims[j] = temp_pims

        for index, p in enumerate(pims.keys()):
            for seg in pims[p].keys():
                pims[p][seg][0] = [' '.join(text for text in segment['originalText']) for segment in self.segments_list if segment['id'] == pims[p][seg][3]]
                inverse_dangling_pims.append(pims[p][seg][3])

        # c_len = 0
        # for segment in self.segments_list:
        #    if segment['id'] not in inverse_dangling_pims:
        #        while c_len in pims.keys():
        #            c_len += 1
        #        pims[c_len] = {"segment0": [' '.join(text for text in segment['originalText']), segment['startTime'], segment['spokenBy'], segment['id']]}

        new_pim = {}
        for pim in list(pims.keys()):
            seen = []
            new_pim[pim] = {}
            index = 0
            for seg in list(pims[pim]):
                if pims[pim][seg][3] in seen:
                    pass
                else:
                    new_pim[pim]['segment' + str(index)] = {}
                    new_pim[pim]['segment' + str(index)] = pims[pim][seg]
                    index += 1
                    seen.append(pims[pim][seg][3])

        return new_pim

    def wrap_community_by_time_refined(self, pims):
        # Add segments which were dangling. 
#         c_len = 0
#         for segment in self.segments_org['segments']:
#             if (segment['id'] not in self.segments_order.keys()):
#                 while c_len in pims.keys():
#                     c_len += 1
#                 pims[c_len] = {"segment0": [' '.join(text for text in segment['originalText']), segment['startTime'], segment['spokenBy'], segment['id']]}

        inverse_dangling_pims = []
        pims_keys = list(pims.keys())
        i = 0
        j = 0
        while i != len(pims_keys):
            j = 0
            while j != len(pims_keys):
                if i != j and pims_keys[i] in pims and pims_keys[j] in pims and (len(pims[pims_keys[i]]) != 1 or len(pims[pims_keys[j]]) != 1):
                    if (pims[pims_keys[i]]['segment0'][1] >= pims[pims_keys[j]]['segment0'][1] and pims[pims_keys[i]]['segment0'][1] <= pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()) - 1)][1]) and (pims[pims_keys[i]]['segment' + str(len(pims[pims_keys[i]].values()) - 1)][1] >= pims[pims_keys[j]]['segment0'][1] and pims[pims_keys[i]]['segment' + str(len(pims[pims_keys[i]].values()) - 1)][1] <= pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()) - 1)][1]):
                        for seg in pims[pims_keys[i]].values():
                            pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()))] = seg
                        del pims[pims_keys[i]]

                        sorted_j = sorted(pims[pims_keys[j]].values(), key=lambda kv: kv[1], reverse=False)
                        temp_pims = {}
                        new_index = 0
                        for new_seg in sorted_j:
                            temp_pims['segment' + str(new_index)] = new_seg
                            new_index += 1
                        pims[pims_keys[j]] = temp_pims
                        j = -1
                        i = 0
                #     elif (pims[pims_keys[i]]['segment0'][1] >= pims[pims_keys[j]]['segment0'][1] and pims[pims_keys[i]]['segment0'][1] <= pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()) - 1)][1]) and (pims[pims_keys[i]]['segment' + str(len(pims[pims_keys[i]].values()) - 1)][1] >= pims[pims_keys[j]]['segment0'][1] and pims[pims_keys[i]]['segment' + str(len(pims[pims_keys[i]].values()) - 1)][1] >= pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()) - 1)][1]):

                #         for seg in pims[pims_keys[i]].values():
                #             pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()))] = seg
                #         del pims[pims_keys[i]]

                #         sorted_j = sorted(pims[pims_keys[j]].values(), key=lambda kv: kv[1], reverse=False)
                #         temp_pims = {}
                #         new_index = 0
                #         for new_seg in sorted_j:
                #             temp_pims['segment' + str(new_index)] = new_seg
                #             new_index += 1
                #         pims[pims_keys[j]] = temp_pims
                #         j = -1
                #         i = 0
                #     elif (pims[pims_keys[i]]['segment0'][1] <= pims[pims_keys[j]]['segment0'][1] and pims[pims_keys[i]]['segment0'][1] <= pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()) - 1)][1]) and (pims[pims_keys[i]]['segment' + str(len(pims[pims_keys[i]].values()) - 1)][1] >= pims[pims_keys[j]]['segment0'][1] and pims[pims_keys[i]]['segment' + str(len(pims[pims_keys[i]].values()) - 1)][1] <= pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()) - 1)][1]):
                #         for seg in pims[pims_keys[i]].values():
                #             pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()))] = seg
                #         del pims[pims_keys[i]]

                #         sorted_j = sorted(pims[pims_keys[j]].values(), key=lambda kv: kv[1], reverse=False)
                #         temp_pims = {}
                #         new_index = 0
                #         for new_seg in sorted_j:
                #             temp_pims['segment' + str(new_index)] = new_seg
                #             new_index += 1
                #         pims[pims_keys[j]] = temp_pims
                #         j = -1
                #         i = 0
                j += 1
            i += 1
        
        for index, p in enumerate(pims.keys()):
            for seg in pims[p].keys():
                    # pims[p][seg][0] = [' '.join(text for text in segment['originalText']) for segment in self.segments_list if segment['id'] == pims[p][seg][3]]
                    pims[p][seg][0] = [segment['originalText'] for segment in self.segments_org["segments"] if segment['id'] == pims[p][seg][3]]
                    if (pims[p].keys())!=1:
                        inverse_dangling_pims.append(pims[p][seg][3])

#         # Add segments which were dangling. 
#         c_len = 0
#         for segment in self.segments_org:
#             if (segment['id'] not in inverse_dangling_pims):
#                 while c_len in pims.keys():
#                     c_len += 1
#                 pims[c_len] = {"segment0": [' '.join(text for text in segment['originalText']), segment['startTime'], segment['spokenBy'], segment['id']]}

        
#         for segmentid in self.segments_order.keys():
#             if (segmentid not in inverse_dangling_pims):
#                 order = self.segments_order[segmentid]
#                 for pim in pims.keys():
#                     if len(pims[pim].keys())!=1:
#                         print(self.segments_order[pims[pim]['segment' + str(len(pims[pim].values()) - 1)][-1]], order - 1)
#                         if self.segments_order[pims[pim]['segment' + str(len(pims[pim].values()) - 1)][-1]] in [order - 1, order -2]:
#                             print ("appending extra segment based on order: ", self.segments_map[segmentid], pim )
#                             pims[pim]['segment' + str(len(pims[pim].values()))] = (self.segments_map[segmentid]['originalText'], self.segments_map[segmentid]['spokenBy'], self.segments_map[segmentid]['startTime'], self.segments_map[segmentid]['id'])
#                             break
        
        
                        
        # Remove Redundent PIMs in a group and also for single segment as a topic accept it as a topic only if it has duration greater than 30 sec.
        new_pim = {}
        track_single_seg = []
        for pim in list(pims.keys()):
            #if len(pims[pim]) == 1:
            #    track_single_seg.append(pims[pim]["segment0"][3])
            seen = []
            new_pim[pim] = {}
            index = 0
            for seg in list(pims[pim]):
                if pims[pim][seg][3] in seen:
                    pass
                else:
                    new_pim[pim]['segment' + str(index)] = {}
                    new_pim[pim]['segment' + str(index)] = pims[pim][seg]
                    index += 1
                    seen.append(pims[pim][seg][3])

        # Remove Redundent PIMs in a group and also for single segment as a topic accept it as a topic only if the word count is greater than 120.
        flag = False
        index = 0
        for pim in list(new_pim.keys()):
            if len(new_pim[pim]) > 1:
                flag = True
        if not flag:
            return new_pim

        index = 0
        for pim in list(new_pim.keys()):
            if len(new_pim[pim]) == 1:
                if (
                    len(
                        self.segments_map[new_pim[pim]["segment0"][-1]][
                            "originalText"
                        ].split(" ")
                    )
                    < 40
                ):
                    del new_pim[pim]
        return new_pim
    
    def order_groups_by_score(self, pims, fv_mapped_score):
        new_pims = {}
        group_score_mapping = {}
        for key in list(pims.keys()):
            group_score = []
            for segi in pims[key].keys():
                if pims[key][segi][3] in fv_mapped_score.keys():
                    group_score.append(fv_mapped_score[pims[key][segi][3]])
            if group_score == []:
                group_score = 0
            group_score_mapping[key] = np.mean(group_score)

        sorted_groups = sorted(group_score_mapping.items(), key=lambda kv: kv[1], reverse=True)
        index = 0
        for groupid, score in sorted_groups:
            new_pims[index] = pims[groupid]
            # new_pims[index]['distance'] = score
            index+=1
        return new_pims

    def combine_pims_by_time(self, pims, group_info):
        print("Before Merging", len(pims.keys()))
        sorted_pims = sorted(pims.items(), key=lambda kv: kv[1]["segment0"][1], reverse=False)
        new_pims = {}
        merge_group = []
#         for pos, (index, pim) in enumerate(sorted_pims):
#             for pos1 ,(index1, pim1) in enumerate(sorted_pims):
#                 if index != index1 and pos1 - pos == 1:
#                     if self.segments_order[pim["segment" + str(len(pim.keys())-1)][-1]] - self.segments_order[pim1["segment0"][-1]] != -1:
#                         merge_group.append((index, index1))
        for pos, (index, pim) in enumerate(sorted_pims):
            for pos1 ,(index1, pim1) in enumerate(sorted_pims):
                if index != index1 and pos1 - pos == 1:
                    if group_info[pim["segment" + str(len(pim.keys())-1)][-1]] == group_info[pim1["segment0"][-1]]:
                        merge_group.append((index, index1))
        print (merge_group)
        for group in [i for j in merge_group for i in j]:
            print ("\n")
            print (*[" ".join(i[0]) for i in pims[group].values()], sep="\n")
        tracking_changes = {}
        pim_seg = {}
        for group1, group2 in merge_group:
            seg_update = []
            if group1 in tracking_changes.keys():
                already_appended_group = tracking_changes[group1]
                for seg in list(pims[group2].values()):
                    seg_update.append(seg)
                tracking_changes[group2] = already_appended_group
                pim_seg[already_appended_group].append(seg_update)
            else:
                for seg in list(pims[group1].values()) + list(pims[group2].values()):
                    seg_update.append(seg)
                tracking_changes[group2] = group1
                if group1 in pim_seg.keys():
                    pim_seg[group1].append(seg_update)
                else:
                    pim_seg[group1] = [seg_update]

        for index, (groupno, group) in enumerate(pim_seg.items()):
            index_n = 0
            new_pims[index] = {}
            for seg in [i for j in group for i in j]:
                new_pims[index]["segment" + str(index_n)] = seg
                index_n += 1
        
        inverse_merge_group = [group for group in pims.keys() if group not in [i for j in merge_group for i in j]]
        index = len(new_pims)
        
        for group in inverse_merge_group:
            new_pims[index] = pims[group]
            index +=1
        print("After Merging", len(new_pims.keys()))
        return new_pims
    
    def h_communities(self, h_flag = False):
        fv, graph_list, fv_mapped_score = self.compute_feature_vector_gpt(local=True)
        #_ = self.remove_preprocessed_segments(graph_list)
        #meeting_graph, yetto_prune = self.construct_graph_ns(fv, graph_list)
        meeting_graph, yetto_prune = self.construct_graph_ns_max(fv, graph_list)
        #meeting_graph, yetto_prune = self.construct_graph_vanilla(fv, graph_list)
        #import pickle
        #with open("meeting_graph", "wb") as f:
        #    pickle.dump([meeting_graph.nodes(data=True), meeting_graph.edges(data=True), graph_list], f)
        v = 0
        t = 1
        edge_count = meeting_graph.number_of_edges()
        meeting_graph_pruned = self.prune_edges_outlier(meeting_graph, graph_list, yetto_prune, v)
        
        l_mod = 1
        flag = False
        community_set = None
        for itr in range(5):
            cs = community.best_partition(meeting_graph_pruned, resolution=t)
            mod = community.modularity(cs, meeting_graph_pruned)
            if mod<l_mod:
                l_mod = mod
                community_set = cs
                flag = True
                
        if not flag:
            community_set = cs
        logger.info("Meeting Graph results", extra={"edges before prunning": edge_count, "edges after prunning": meeting_graph_pruned.number_of_edges(), "modularity": mod})    
        community_set_sorted = sorted(community_set.items(), key=lambda kv: kv[1], reverse=False)
        clusters = []
        temp = []
        prev_com = 0
        for index,(word,cluster) in enumerate(community_set_sorted):
            if prev_com==cluster:
                temp.append(word)
                if index==len(community_set_sorted)-1:
                    clusters.append(temp)
            else:
                clusters.append(temp)
                temp = []
                prev_com = cluster
                temp.append(word)
        if (h_flag):
            v = 75
            community_set_collection = []
            old_cluster = []
            # print (clusters)
            for cluster in clusters:
                if len(cluster) >= 2:
                    graph_list_pruned = copy.deepcopy(graph_list)
                    for k in graph_list.keys():
                        if k not in cluster:
                            del graph_list_pruned[k]

                    meeting_graph, yetto_prune = self.construct_graph(fv, graph_list_pruned)
                    meeting_graph_pruned = self.prune_edges_outlier(meeting_graph, graph_list_pruned, yetto_prune, v)
                    community_set = community.best_partition(meeting_graph_pruned)
                    community_set_sorted = sorted(community_set.items(), key=lambda kv: kv[1], reverse=False)
                    #print (community_set_sorted)
                    i = 0
                    prev_cluster = 9999999999999999
                    for (sent, cls) in community_set_sorted:
                        if cls not in old_cluster:
                            community_set_collection.append((sent, cls))
                            old_cluster.append(cls)
                            prev_cluster = cls
                            i = cls
                        else:
                            if cls == prev_cluster:
                                community_set_collection.append((sent, i))
                                continue
                            while i in old_cluster:
                                i += 1
                            prev_cluster = cls
                            community_set_collection.append((sent, i))
                            old_cluster.append(i)
                    for (sent, cls) in community_set_sorted:
                        old_cluster.append(cls)
                else:
                    i = 0
                    while i in old_cluster:
                        i += 1
                    community_set_collection.append((cluster[0], i))
                    old_cluster.append(i)
                # print(community_set_collection)
#             result = []
#             temp = []
#             prev = 0
#             for sent, cls in community_set_collection:
#                 if prev != cls:
#                     result.append(temp)
#                     temp = []
#                     print ("cluster -=======> ", cls)
#                     prev = cls
#                 else:
#                     temp.append(graph_list[sent][0])
#                 print (graph_list[sent][0])
#             result.append(temp)
#             import pickle
#             with open("results","wb") as f:
#                 pickle.dump(result, f)
            community_set_collection = sorted(community_set_collection, key = lambda x: x[1], reverse=False)
            #print (community_set_collection)
            community_timerange = self.refine_community(community_set_collection, graph_list)
            #print (community_timerange)
            # logger.info("commnity timerange", extra={"timerange": community_timerange})
            pims = self.group_community_by_time(community_timerange)
            pims = self.wrap_community_by_time_refined(pims)
            
            logger.info("Final PIMs", extra={"PIMs": pims})
        else:
            community_set_collection = deepcopy(community_set_sorted)
            community_set_collection = sorted(community_set_collection, key = lambda x: x[1], reverse=False)
            community_timerange = self.refine_community(community_set_collection, graph_list)
            # print (community_timerange)
            # logger.info("commnity timerange", extra={"timerange": community_timerange})
            # _ = self.remove_preprocessed_segments(graph_list)
            pims = self.group_community_by_time(community_timerange)
            pims = self.wrap_community_by_time_refined(pims)
            pims = self.order_groups_by_score(pims, fv_mapped_score)
#             rw = nx.Graph()
#             for key in pims.keys():
#                 for segkey in pims[key]:
#                     rw.add_node(pims[key][segkey][-1], text = pims[key][segkey][0], speaker = pims[key][segkey][2])
                
#                 for nodea in rw.nodes.data():
#                     for nodeb in rw.nodes.data():
#                         rw.add_edge(nodea, nodeb)
                        
#             with open("meeting_graph", "wb") as f:
#                 pickle.dump([rw.nodes(data=True), rw.edges(data=True)], f)
            logger.info("Final PIMs", extra={"PIMs": pims})
        return pims
    
    def construct_graph_para_new(self, meeting_graph):
        import statistics
        max_connection = {}
        max_score = {}
        outlier_score = {}
        for node in meeting_graph.nodes():
            closest_connection_n = sorted(dict(meeting_graph[node]).items(), key=lambda kv:kv[1]["weight"], reverse=True)
            
            max_score_current = []
            weights_n = list(map(lambda kv: (kv[1]["weight"]).tolist(), closest_connection_n))
            q3 = np.percentile(weights_n, 75)
            iqr = np.subtract(*np.percentile(weights_n, [75, 25]))
            outlier_score[node] = {}
            outlier_score[node]["outlier"] = q3 + 1 * iqr
            outlier_score[node]["iqr"] = iqr
            outlier_score[node]["q3"] = q3
            outlier_score[node]["weights_n"] = closest_connection_n
            outlier_score[node]["avg+pstd"] = statistics.mean(weights_n)+statistics.pstdev(weights_n)


        graph_data = deepcopy(meeting_graph.edges.data())
        for nodea, nodeb, weight in graph_data:
            if weight["weight"] > outlier_score[nodea]["outlier"]:
                pass
            else:
                meeting_graph.remove_edge(nodea, nodeb)
                
        return meeting_graph
    
    def itr_communities(self):
        fv, graph_list = self.compute_feature_vector_gpt(local=True)
        meeting_graph, yetto_prune = self.construct_graph_ns_max(fv, graph_list)
        v = 0
        t = 1.0
        edge_count = meeting_graph.number_of_edges()
        meeting_graph_pruned = self.prune_edges_outlier(meeting_graph, graph_list, yetto_prune, v)
        
        l_mod = 1
        flag = False
        community_set = None
        print ("Using Community Algorithm")
        for itr in range(5):
            cs = community.best_partition(meeting_graph_pruned, resolution=t)
            mod = community.modularity(cs, meeting_graph_pruned)
            if mod<l_mod:
                l_mod = mod
                community_set = cs
                flag = True
                
        if not flag:
            community_set = cs
########################
        

#         chinese_whispers(meeting_graph_pruned, weighting='nolog', iterations=20)
#         community_set = {}
#         for node in meeting_graph_pruned.nodes():
#             community_set[node]=meeting_graph_pruned.nodes[node]['label']
            

########################
        #logger.info("Meeting Graph results", extra={"edges before prunning": edge_count, "edges after prunning": meeting_graph_pruned.number_of_edges(), "modularity": mod})    
        community_set_sorted = sorted(community_set.items(), key=lambda kv: kv[1], reverse=False)
        community_set_collection = deepcopy(community_set_sorted)
        community_set_collection = sorted(community_set_collection, key = lambda x: x[1], reverse=False)
        community_timerange = self.refine_community(community_set_collection, graph_list)
        pims = self.group_community_by_time(community_timerange)
        pims = self.wrap_community_by_time_refined(pims)
        #pims = self.order_groups_by_score(pims, fv_mapped_score)
        
        #pims = self.combine_pims_by_time(pims)
        #logger.info("Final PIMs", extra={"PIMs": pims})
        graph_list_index = {}
        for index, g in enumerate(graph_list.values()):
            
            if g[-1] not in graph_list_index.keys():
                graph_list_index[g[-1]] = [index]
            else:
                graph_list_index[g[-1]].append(index)
        group = []
        pim_fv = []
        for pim in pims.keys():
            pim_seg = []
            fv_index= []
            for seg in pims[pim].keys():
                pim_seg.append(pims[pim][seg])
                fv_index.append([fv[x] for x  in graph_list_index[pims[pim][seg][-1]]])
            group.append(pim_seg)
            pim_fv.append(np.mean([ i for j in fv_index for i in j], axis=0))
        
        G2 = nx.Graph()
        for index1 in range(len(pim_fv)):
            for index2 in range(len(pim_fv)):
                G2.add_edge(index1, index2, weight = cosine(pim_fv[index1], pim_fv[index2]))
        
        
        import pickle
        with open("para_graph", "wb") as f:
            pickle.dump([G2.nodes(data=True), G2.edges(data=True), group], f)
        #G3 = deepcopy(self.construct_graph_para_new(G2))
        cs2 = community.best_partition(G2, resolution=1.0)
        cs2_sorted = sorted(cs2.items(), key = lambda x: x[1], reverse=False)
        print (cs2_sorted)
        prev = 0
        group_seg_list = {}
        for seg, cluster in cs2_sorted:
            if prev !=cluster:
                print ("\n\n====================Group Cluster=========================")
                prev=cluster
            print(group[seg], "\n\n")
            for segi in list(map(lambda kv: kv[-1], group[seg])):
                group_seg_list[segi] = cluster
        print (prev)
        pims = self.combine_pims_by_time(pims, group_seg_list)
        return pims

    
    
    def get_communities(self):
        fv, graph_list, fv_mapped_score = self.compute_feature_vector_gpt(local=False)
        meeting_graph, yetto_prune = self.construct_graph(fv, graph_list)
        v = 0
        edge_count = meeting_graph.number_of_edges()
        meeting_graph_pruned = self.prune_edges_outlier(meeting_graph, graph_list, yetto_prune, v)
        #nx.write_gexf(meeting_graph, "meeting_graph.gexf")
        import pickle
        with open("meeting_graph", "wb") as f:
            pickle.dump([meeting_graph_pruned.nodes(data=True), meeting_graph_pruned.edges(data=True), graph_list], f)
        community_set = community.best_partition(meeting_graph_pruned)
        mod = community.modularity(community_set, meeting_graph_pruned)
        logger.info("Meeting Graph results", extra={"edges before prunning": edge_count, "edges after prunning": meeting_graph_pruned.number_of_edges(), "modularity": mod})    
        community_set_sorted = self.compute_louvian_community(meeting_graph_pruned, community_set)
        community_timerange = self.refine_community(community_set_sorted, graph_list)
        # logger.info("commnity timerange", extra={"timerange": community_timerange})
        pims = self.group_community_by_time(community_timerange)
        pims = self.wrap_community_by_time_refined(pims)
        logger.info("Final PIMs", extra={"PIMs": pims})
        return pims


    def get_communities_prune(self):
        print("Computing Community w.r.t pruning using modularity")
        fv, graph_list, fv_mapped_score = self.compute_feature_vector_gpt(local=False)
        meeting_graph, yetto_prune = self.construct_graph(fv, graph_list)
        max_mod = 0
        min_mod = 1
        v = 50
        while True:
            meeting_graph_pruned = self.prune_edges_outlier(meeting_graph, graph_list, yetto_prune, v)
            community_set = community.best_partition(meeting_graph_pruned)
            mod = community.modularity(community_set, meeting_graph_pruned)
            logger.info("Meeting Graph results", extra={"edges before prunning": meeting_graph.number_of_edges(), "edges after prunning": meeting_graph_pruned.number_of_edges(), "modularity ": mod})
            if mod > max_mod and mod <= 0.40:
                max_meeting_grap_pruned = meeting_graph_pruned
                max_community_set = community_set
                max_mod = mod
                flag = True
            elif mod > 0.40 and not flag:
                max_meeting_grap_pruned = meeting_graph_pruned
                max_community_set = community_set
                break
            elif mod > 0.40:
                break
            if mod < min_mod:
                min_mod = mod
            v +=1
        meeting_graph_pruned = max_meeting_grap_pruned
        community_set = max_community_set
        if flag:
            mod = max_mod
        else:
            mod = min_mod

        logger.info("Meeting Graph results", extra={"edges before prunning": meeting_graph.number_of_edges(), "edges after prunning": meeting_graph_pruned.number_of_edges(), "modularity": mod})
        community_set_sorted = sorted(community_set.items(), key=lambda kv: kv[1], reverse=False)
        # nx.write_gexf(meeting_graph_pruned, "meeting_graph_pruned.gexf")
        import pickle
        with open("meeting_graph", "wb") as f:
            pickle.dump([meeting_graph_pruned.nodes(data=True), meeting_graph_pruned.edges(data=True), graph_list], f)
        community_timerange = self.refine_community(community_set_sorted, graph_list)
        pims = self.group_community_by_time(community_timerange)
        pims = self.wrap_community_by_time_refined(pims)
        pims = self.order_groups_by_score(pims, fv_mapped_score)
        logger.info("Final PIMs", extra={"PIMs": pims})
        return pims




