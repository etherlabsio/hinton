import sys
sys.path.append("../")
sys.path.append("/home/ray__/ssd/BERT/")
sys.path.append("/home/ray__/CS/org/etherlabs/ai-engine/pkg/")
import text_preprocessing.preprocess as tp
from extra_preprocess import preprocess_text
from filter_groups import CandidateKPExtractor
import nltk
import networkx as nx
from gpt_feat_utils import GPT_Inference
from scipy.spatial.distance import cosine
import numpy as np
from community import best_partition
from copy import deepcopy
#gpt_model = GPT_Inference("/home/ray__/ssd/BERT/models/ai/epoch3/", device="cuda")
#gpt_model = GPT_Inference("/home/ray__/ssd/BERT/models/se/epoch3/", device="cpu")
gpt_model = GPT_Inference("/home/ray__/ssd/BERT/models/customer_service/epoch3/", device="cuda")
#gpt_model = GPT_Inference("/home/ray__/ssd/BERT/models/product/", device="cuda")
#gpt_model = GPT_Inference("/home/ray__/ssd/BERT/models/ether_v2/ether_googleJan13_groupsplit_withstop_4+w_gt3s_lr3e-5/",device="cpu")

sys.path.append("../helper_functions/")

from get_groups import call_gs 
import sys
sys.path.append("../")
from filter_groups import CandidateKPExtractor



def get_ent(request, ent_fv, com_map, kp_entity_graph):
    kp_e = CandidateKPExtractor()
    uncased_nodes = [ele.lower() for ele in kp_entity_graph]
    uncased_node_dict = dict(zip(list(kp_entity_graph),uncased_nodes))

    group = call_gs(request)
    group_ent = {}
    for groupid, groupobj in group.items():
        seg_text = " ".join([segobj['originalText'] for segobj in groupobj])
        text_kps = kp_e.get_candidate_phrases(seg_text)
        text_kps = list(set([ele.lower() for ele in text_kps]))
        tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(seg_text))
        text_nouns = []
        for tagged_sent in tagged_sents:
            text_nouns.extend([ele[0] for ele in list(tagged_sent) if ele[1].startswith('NN')])
        text_nouns = [ele.lower() for ele in text_nouns]
        intersecting_nouns = list(set(text_nouns)&set(kp_entity_graph))
        intersection_ctr = 0
        filtered_kps = []
        for kp in text_kps:
            if len(kp.split(' '))>1:
                kp_nouns = list(set(kp.split(' '))&set(intersecting_nouns))
#                 for noun in kp_nouns:
#                     rem_nouns = list(set(kp_nouns)-set([noun]))
#                     if set(rem_nouns)&set(kp_entity_graph[noun])==set(rem_nouns):
#                         filtered_kps.append(kp)
#                         continue
                for noun in kp_nouns:
                    if noun in kp_entity_graph.nodes():
                        filtered_kps.append(kp)
                        continue
        filtered_kps = list(set(filtered_kps))
        candidate_sents = [sent.lower() for sent in nltk.sent_tokenize(seg_text)]
        filtered_sents = []
        for sent in candidate_sents:
            if any(kp in sent for kp in filtered_kps):
                filtered_sents.append(sent)
        noun_list = [ele.split(' ') for ele in filtered_kps]
        noun_list = sum(noun_list, [])
        noun_list = list(set(noun_list)&set([uncased_node_dict[ele] for ele in uncased_node_dict]))
        noun_node_list = [key  for (key, value) in uncased_node_dict.items() if value in noun_list]
        ent_node_list = [ele for ele in noun_node_list if kp_entity_graph.nodes[ele]['node_type']=='entity']
        noun_node_list = list(set(noun_node_list)-set(ent_node_list))

        kp_Map_list = []
        kp_ent_map = []
        for noun in noun_node_list:
            kp_Map_list.extend([ele for ele in list(kp_entity_graph[noun]) 
                                if kp_entity_graph[noun][ele]['edge_type']=='token_kp_map'])
        
        for kp in list(set(kp_Map_list)):
            kp_ent_map.extend([ele for ele in list(kp_entity_graph[kp]) if kp_entity_graph.nodes[ele]['node_type']=='entity'])

        kp_ent_map_intrm = deepcopy(kp_ent_map)
        for ent in kp_ent_map_intrm:
            if kp_entity_graph.nodes[ent]['is_ether_node']==True:
                kp_ent_map.append("<ETHER>-"+ent)
                
        kp_ent_map = list(set(kp_ent_map+ent_node_list))
        kp_ent_map = list(set(kp_ent_map)&set(ent_fv))

        sent_list = filtered_sents
        sent_fv = [gpt_model.get_text_feats(sent) for sent in sent_list]
        G = nx.Graph()
        G.add_nodes_from(range(len(sent_fv)))


        node_list = range(len(sent_fv))
        for index1, nodea in enumerate(range(len(sent_fv))):
            for index2, nodeb in enumerate(range(len(sent_fv))):
                if index2 >= index1:
                    c_score = 1 - cosine(sent_fv[nodea], sent_fv[nodeb])
                    #if c_score>= outlier_score:
                    G.add_edge(nodea, nodeb, weight = c_score)
            closest_connection_n = sorted(dict(G[nodea]).items(), key=lambda kv:kv[1]["weight"], reverse=True)
            weights_n = list(map(lambda kv: (kv[1]["weight"]).tolist(), closest_connection_n))
            q3 = np.percentile(weights_n, 75)
            iqr = np.subtract(*np.percentile(weights_n, [75, 25]))
            #outlier_score = q3 + (1.5 * iqr)
            outlier_score = q3 + (1 * iqr)
            for nodeb, param in dict(G[nodea]).items():
                if param['weight']>=q3:
                    pass
                else:
                    G.remove_edge(nodea, nodeb)




        comm_temp = best_partition(G, resolution=1)

        prev = 0 
        comm_map = {}
        for ent, cls in sorted(comm_temp.items(),key=lambda kv:kv[1]):
            if prev!=cls:
                prev = cls
            if cls in comm_map.keys():
                comm_map[cls].append(ent)
            else:
                comm_map[cls] = [ent]

        agg_fv = {}
        if True in [True if len(s_list)>1 else False for s_list in comm_map.values() ]:
            threshold = 1
        else:
            threshold = 0
        for comm, s_list in comm_map.items():
            if len(s_list)>threshold:
                temp_fv = [sent_fv[s] for s in s_list]
                agg_fv[comm] = np.mean(temp_fv, axis=0)

        dist_list = {}
        for pos, fv in agg_fv.items():
            temp_list = []
            for entity in ent_fv.keys():
                temp_list.append((entity, 1-cosine(ent_fv[entity], fv)))
            dist_list[pos] = sorted(temp_list, key=lambda kv:kv[1], reverse=True)[:10]

        group_ent[groupid] = [e for e_list in dist_list.values() for e in e_list]
   
    return group, group_ent