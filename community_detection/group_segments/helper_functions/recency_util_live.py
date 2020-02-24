import sys
sys.path.append("/home/ray__/ssd/BERT/") # gpt model utils location
from gpt_feat_utils import GPT_Inference
import json

from copy import deepcopy
import pickle, json
import numpy as np
from scipy.spatial.distance import cosine
from collections import Counter

sys.path.append("../") # grouping functions main.py file location
sys.path.append("/home/ray__/CS/org/etherlabs/ai-engine/pkg/") # ai-engine pkg location


import text_preprocessing.preprocess as tp
from extra_preprocess import preprocess_text
from generate_controlled_top_10_ent import get_ent
  
    

def get_artifacts(artifacts_dir):
    ent_fv_full = pickle.load(open(artifacts_dir + "entity.pkl","rb"))
    sent_dict = pickle.load(open(artifacts_dir + "sent_dict.pkl", "rb"))
    com_map = pickle.load(open(artifacts_dir + "com_map.pkl", "rb"))
    kp_entity_graph = pickle.load(open(artifacts_dir + "kp_entity_graph.pkl", "rb"))
    gc = pickle.load(open(artifacts_dir + "gc.pkl", "rb"))
    lc = pickle.load(open(artifacts_dir + "lc.pkl", "rb"))
    common_entities = ent_fv_full.keys() & com_map.keys()
    ent_fv = {}
    for ent in common_entities:
        ent_fv[ent] = ent_fv_full[ent]
        
    return ent_fv, kp_entity_graph, com_map, gc, lc

def get_ranked_groups(request, ent_fv, com_map, kp_entity_graph):
    group, group_ent = get_ent(request, ent_fv, com_map, kp_entity_graph)
    return group, group_ent

def get_entity_mapping(group_ent, ent_fv, com_map, kp_entity_graph):
    group_ent_map = {}
    for groupid, ent_list in group_ent.items():
        group_ent_map[groupid] = [com_map[ent] for ent in list(map(lambda kv:kv[0], group_ent[groupid]))]
        #group_ent_score[groupid] = [ranked_com[com] for com in group_ent_score[groupid] if com in ranked_com.keys()]
    
    return group_ent_map

def filter_entity_community(group_ent_map, ent_fv, com_map, kp_entity_graph):
    group_ent_map_filtered_intrm = {}
    group_ent_map_filtered = {}
    for groupid, ent_map in group_ent_map.items():
        filtered_ent_map = []
        if len(set(ent_map)) == len(ent_map):
            group_ent_map_filtered[groupid] = []
        else:
            count_a = Counter(ent_map).most_common()
            for i, count in count_a:
                if count>1 :
                    filtered_ent_map.append((i, count))

            group_ent_map_filtered[groupid] = filtered_ent_map

    return group_ent_map_filtered


def get_entity_mapping_rank(group_ent_map_filtered, gc, lc, ent_fv, com_map, kp_entity_graph):
    print (group_ent_map_filtered)
    group_ent_map_rank_lc = {}
    group_ent_map_rank_gc = {}
    for groupid, ent_map_list in group_ent_map_filtered.items():
        group_ent_map_rank_intrm_lc = []
        group_ent_map_rank_intrm_gc = []
        
        for ent_map, count in ent_map_list:
            if ent_map in lc.keys() and sum(lc[ent_map])!=0:
                group_ent_map_rank_intrm_lc.append(sum(lc[ent_map]))
            else:
                if ent_map in gc.keys():
                    group_ent_map_rank_intrm_gc.append(gc[ent_map])
                else:
                    group_ent_map_rank_intrm_gc.append(0)
        if group_ent_map_rank_intrm_lc!=[]:
            group_ent_map_rank_lc[groupid] = sum(group_ent_map_rank_intrm_lc)
        else:
            group_ent_map_rank_gc[groupid] = sum(group_ent_map_rank_intrm_gc) 
        
    ## update gc and lc
    updated_lc_list = []
    updated_comm_list = []
    for groupid, ent_map_list in group_ent_map_filtered.items():
        for ent_map, count in ent_map_list:
            if ent_map in lc.keys():
                if len(lc[ent_map])!=5:
                    if ent_map not in updated_comm_list:
                        lc[ent_map].append(count)
                    else:
                        lc[ent_map].append(lc[ent_map].pop()+count)
                else:
                    if ent_map not in updated_comm_list:
                        del lc[ent_map][0]
                        lc[ent_map].append(count)
                    else:
                        lc[ent_map].append(lc[ent_map].pop()+count)
        
                updated_lc_list.append(ent_map)
            else:
                lc[ent_map] = [count]
                updated_lc_list.append(ent_map)
                
            if ent_map in gc.keys():
                gc[ent_map] +=count
            else:
                gc[ent_map] = count
            updated_comm_list.append(ent_map)
    
    lc_copy = deepcopy(list(lc.items()))            
    for ent, freq in lc_copy:
        if ent not in updated_lc_list:
            if sum(lc[ent]) == 0:
                del lc[ent]
            else:
                if len(lc[ent])!=5:
                    lc[ent].append(0)
                else:
                    del lc[ent][0]
                    lc[ent].append(0)
    
    return group_ent_map_rank_lc, group_ent_map_rank_gc, gc, lc 


def compute_groups_new_call(req, artifacts_dir, store=False):
    ent_fv, kp_entity_graph, com_map, gc, lc = get_artifacts(artifacts_dir)
    group, group_ent = get_ranked_groups(req, ent_fv, com_map, kp_entity_graph)
    group_ent_map = get_entity_mapping(group_ent, ent_fv, com_map, kp_entity_graph)
    group_ent_map_filtered = filter_entity_community(group_ent_map, ent_fv, com_map, kp_entity_graph)
    group_ent_map_rank_lc, group_ent_map_rank_gc, gc_copy, lc_copy = get_entity_mapping_rank(group_ent_map_filtered, gc, lc, ent_fv, com_map, kp_entity_graph)
    #print (lc_copy)
    if store:
        print ("writing the gc and lc update.")
        pickle.dump(gc_copy, open(artifacts_dir + "gc.pkl","wb"))
        pickle.dump(lc_copy, open(artifacts_dir + "lc.pkl","wb"))
    return group, group_ent_map_rank_lc, group_ent_map_rank_gc, gc_copy ,lc_copy