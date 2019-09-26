import json
from dataclasses import dataclass, asdict
from typing import List
import logging
from copy import deepcopy


@dataclass
class Request:
    pim_result: list
    gs_result: dict
    gs_rec_map: dict
    pim_rec_map: dict

def decode_json_request(req) -> Request:

    if isinstance(req, str):
        req = json.load(req)

    gs_result = deepcopy(req['groups'])
    pim_result = deepcopy(req['pims'])
    gs_rec_map = {}
    pim_rec_map = {}

    for keys in gs_result.keys():
        for seg in gs_result[keys]:
            gs_rec_map[seg['recordingId']] = keys

    for seg in pim_result["segments"]:
        pim_rec_map[seg["recordingId"]] =  seg['distance']

    return Request(pim_result, gs_result, gs_rec_map, pim_rec_map)
