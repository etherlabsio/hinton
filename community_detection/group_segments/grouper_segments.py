import json
import sys
sys.path.append("../../../ai-engine/pkg/")
import text_preprocessing.preprocess as tp
import extra_preprocess
# from group_segments.extra_preprocess import format_time
import networkx as nx
import math
from scorer import cosine
import community
from datetime import datetime
import scorer
import logging
from log.logger import setup_server_logger
logger = logging.getLogger()


class community_detection():
    segments_list = []
    segments_org = []
    lambda_function = None

    def __init__(self, Request, lambda_function):
        self.segments_list = Request.segments
        self.segments_org = Request.segments_org
        self.lambda_function = lambda_function
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
        transcript_score = scorer.get_feature_vector(mind_input, self.lambda_function)
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
                    
        mind_input = json.dumps({"text": input_list})
        mind_input = json.dumps({"body": mind_input})
        transcript_score = scorer.get_feature_vector(mind_input, self.lambda_function)
        for segment in self.segments_list:
            for sent in segment['originalText']:
                if sent != '':
                    graph_list[index] = (sent, segment['startTime'], segment['spokenBy'], segment['id'])
                    fv[index] = transcript_score[index]
                    index += 1
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
                    
        transcript_score = scorer.get_embeddings(input_list)
        for segment in self.segments_list:
            for sent in segment['originalText']:
                if sent != '':
                    graph_list[index] = (sent, segment['startTime'], segment['spokenBy'], segment['id'])
                    fv[index] = transcript_score[index]
                    index += 1
        return fv, graph_list

    def construct_graph(self, fv, graph_list):
        meeting_graph = nx.Graph()
        yetto_prune = []
        c_weight = 0
        for nodea in graph_list.keys():
            for nodeb in graph_list.keys():
                if nodeb > nodea:
                    c_weight = 1 - cosine(fv[nodea], fv[nodeb])
                    meeting_graph.add_edge(nodea, nodeb, weight=c_weight)
                    yetto_prune.append((nodea, nodeb, c_weight))
        return meeting_graph, yetto_prune

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
        for index, (word, cluster) in enumerate(community_set_sorted):
            if prev_com == cluster:
                temp.append(word)
                if index == len(community_set_sorted) - 1:
                    clusters.append(temp)
            else:
                clusters.append(temp)
                temp = []
                prev_com = cluster
                temp.append(word)
        timerange = []
        temp = []
        for cluster in clusters:
            temp = []
            for sent in cluster:
                # temp.append(graph_list[sent])
                # logger.info("segment values", extra={"segment":self.segments_list})
                temp.append(graph_list[sent])
            if len(temp) != 0:
                temp = list(set(temp))
                temp = sorted(temp, key=lambda kv: kv[1], reverse=False)
                timerange.append(temp)

        return timerange

    def group_community_by_time(self, timerange):
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

            for (index1, (sent1, time1, user1, id1)), (index2, (sent2, time2, user2, id2)) in zip(enumerate(com[0:]), enumerate(com[1:])):
                if id1 != id2:
                    if ((extra_preprocess.format_time(time2, True) - extra_preprocess.format_time(time1, True)).seconds <= 120):
                        if (not flag):
                            pims[index_pim] = {'segment' + str(index_segment): [sent1, time1, user1, id1]}
                            index_segment += 1
                            temp.append((sent1, time1, user1, id1))
                        pims[index_pim]['segment' + str(index_segment)] = [sent2, time2, user2, id2]
                        index_segment += 1
                        temp.append((sent2, time2, user2, id2))
                        flag = True
                    else:
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
                    continue
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
        inverse_dangling_pims = []
        pims_keys = list(pims.keys())
        i = 0
        j = 0
        while i != len(pims_keys):
            j = 0
            while j != len(pims_keys):
                if i != j and pims_keys[i] in pims and pims_keys[j] in pims:
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
                    elif (pims[pims_keys[i]]['segment0'][1] >= pims[pims_keys[j]]['segment0'][1] and pims[pims_keys[i]]['segment0'][1] <= pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()) - 1)][1]) and (pims[pims_keys[i]]['segment' + str(len(pims[pims_keys[i]].values()) - 1)][1] >= pims[pims_keys[j]]['segment0'][1] and pims[pims_keys[i]]['segment' + str(len(pims[pims_keys[i]].values()) - 1)][1] >= pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()) - 1)][1]):

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
                    elif (pims[pims_keys[i]]['segment0'][1] <= pims[pims_keys[j]]['segment0'][1] and pims[pims_keys[i]]['segment0'][1] <= pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()) - 1)][1]) and (pims[pims_keys[i]]['segment' + str(len(pims[pims_keys[i]].values()) - 1)][1] >= pims[pims_keys[j]]['segment0'][1] and pims[pims_keys[i]]['segment' + str(len(pims[pims_keys[i]].values()) - 1)][1] <= pims[pims_keys[j]]['segment' + str(len(pims[pims_keys[j]].values()) - 1)][1]):
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
                j += 1
            i += 1
        for index, p in enumerate(pims.keys()):
            for seg in pims[p].keys():
                # pims[p][seg][0] = [' '.join(text for text in segment['originalText']) for segment in self.segments_list if segment['id'] == pims[p][seg][3]]
                pims[p][seg][0] = [segment['originalText'] for segment in self.segments_org["segments"] if segment['id'] == pims[p][seg][3]]
                inverse_dangling_pims.append(pims[p][seg][3])

        # c_len = 0
        # for segment in self.segments_list:
        #    if (segment['id'] not in inverse_dangling_pims):
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

    def get_communities(self):
        # segments_data = ' '.join([sentence for segment in self.segments_list for sentence in segment['originalText']])
        fv, graph_list = self.compute_feature_vector()
        #fv, graph_list = self.compute_feature_vector_gpt()
        #fv, graph_list = self.compute_feature_vector_use()
        logger.info("No of sentences is", extra={"sentence": len(fv.keys())})
        meeting_graph, yetto_prune = self.construct_graph(fv, graph_list)
        max_meeting_grap_pruned = None
        max_community_set = None
        max_mod = 0
        for v in [0.15, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01]:
            # flag = False
            for count in range(5):
                meeting_graph_pruned = self.prune_edges(meeting_graph, graph_list, yetto_prune, v)
                community_set = community.best_partition(meeting_graph_pruned)
                mod = community.modularity(community_set, meeting_graph_pruned)
                logger.info("Meeting Graph results", extra={"edges before prunning": meeting_graph.number_of_edges(), "edges after prunning": meeting_graph_pruned.number_of_edges(), "modularity ": mod})
                # if mod>0.3:
                #     flag = True
                #     break
                # if mod==0:
                #     meeting_graph_pruned = self.prune_edges(meeting_graph, graph_list, yetto_prune, 0.15)
                #     flag = True
                #     break
                if mod > max_mod and mod <= 0.36:
                    max_meeting_grap_pruned = meeting_graph_pruned
                    max_community_set = community_set
                    max_mod = mod
                    # flag = True
                    # if flag:
                    #     break
        meeting_graph_pruned = max_meeting_grap_pruned
        community_set = max_community_set
        mod = max_mod

        logger.info("Meeting Graph results", extra={"edges before prunning": meeting_graph.number_of_edges(), "edges after prunning": meeting_graph_pruned.number_of_edges(), "modularity": mod})
        community_set_sorted = self.compute_louvian_community(meeting_graph_pruned, community_set)
        community_timerange = self.refine_community(community_set_sorted, graph_list)
        # logger.info("commnity timerange", extra={"timerange": community_timerange})
        pims = self.group_community_by_time(community_timerange)
        pims = self.wrap_community_by_time_refined(pims)
        logger.info("Final PIMs", extra={"PIMs": pims})
        return pims
