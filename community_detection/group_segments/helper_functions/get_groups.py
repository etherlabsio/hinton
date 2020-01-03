import json
import sys
sys.path.append("../")

from main import handler

def call_gs(request):
    print ("nope")
    res = handler(request, None)
    group = json.loads(res['body'])

    group_sorted = {}
    group_sorted ["group"] = {}
    temp_group = sorted(group['group'].items(), key= lambda kv:kv[1][0]['startTime'], reverse=False)
    for g in temp_group:
        group_sorted["group"][g[0]] = g[1]

    group = group_sorted["group"]
    return group