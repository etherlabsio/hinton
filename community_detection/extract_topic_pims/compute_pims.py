import logging

logger = logging.getLogger()

def get_pims(Request):
    used_topics = []
    group_no = None
    index = 0
    topic_pim = {}
    ranked_pims = sorted([(k,v) for (k,v) in Request.pim_rec_map.items()], key= lambda kv: kv[1])[:10]

    for (rec_id, distance) in ranked_pims:
        if rec_id in Request.gs_rec_map.keys():
            group_no = Request.gs_rec_map[rec_id]

            if group_no not in used_topics and len(Request.gs_result[group_no])!=1:
                topic_pim[index] = group_no
                used_topics.append(group_no)
                index += 1
        if index==6:
            break
    print ("\n", topic_pim)
    final_output = list(map(lambda x: Request.gs_result[x] , topic_pim.values()))
    return final_output
