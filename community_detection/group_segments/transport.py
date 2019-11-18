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

import json
import sys
sys.path.append("../../../ai-engine/pkg/")
from dataclasses import dataclass, asdict
from typing import List
import logging
import text_preprocessing.preprocess as tp
from copy import deepcopy
from extra_preprocess import preprocess_text


@dataclass
class Request:
    mind_id: str
    instance_id: str
    context_id: str
    segments: list
    segments_org: list
    segments_map: dict
    segments_order: dict


def decode_json_request(req) -> Request:

    if isinstance(req, str):
        req = json.load(req)

    def decode_segments(seg):
        segments_text = list(map(lambda x: preprocess_text(x['originalText']), seg['segments']))
        segments_data = deepcopy(seg['segments'])
        assert(len(segments_text ) == len(segments_data))
        for index, segment in enumerate(segments_data):
            segments_data[index]['originalText'] = deepcopy(segments_text[index])
        # segments_map = list(map(lambda x:segments_map[x['id']]=x,seg['segments']))
        return segments_data
    if req['segments'] is None:
        return False

    segments_map = {}
    for segm in req['segments']:
        segments_map[segm['id']] = deepcopy(segm)

    segments_org = deepcopy(req)
    segments = decode_segments(segments_org)

    index = 0
    segments_order = {}
    for ele in sorted(segments, key=lambda x: x['startTime'], reverse=False):
        if ele['originalText'] != "":
            segments_order[ele['id']] = index
            index+=1
    
    return Request(req['mindId'],req['instanceId'],req['contextId'],segments, segments_org, segments_map, segments_order)

