import sys, json
sys.path.append("/home/ray__/ssd/BERT/")
sys.path.append("/home/ray__/CS/org/etherlabs/ai-engine/pkg/")
from gpt_feat_utils import GPT_Inference
gpt_model = GPT_Inference("/home/ray__/ssd/BERT/models/se/epoch3/", device="cpu")

from main import handler

def generate_groups(request):
    res = handler(request, None)
    group = json.loads(res['body'])

    group_sorted = {}
    group_sorted ["group"] = {}
    temp_group = sorted(group['group'].items(), key= lambda kv:kv[1][0]['startTime'], reverse=False)
    for g in temp_group:
        group_sorted["group"][g[0]] = g[1]

    group = group_sorted["group"]
    return group