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
import sys
sys.path.append("../../../ai-engine/pkg")
from grouper import get_groups
import json


from transport import decode_json_request

from extra_preprocess import format_pims_output
import sys
import logging
from log.logger import setup_server_logger
import scorer
import pickle
import os
# from scorer import load_mind_features
import boto3

# +
logger = logging.getLogger()
setup_server_logger(debug=False)

def load_mind_features(mind_id):
    s3 = boto3.resource('s3')
    # BUCKET_NAME = io.etherlabs.artifacts
    bucket = os.getenv('BUCKET_NAME', 'io.etherlabs.artifacts')
    # MINDS = staging2/minds/
    mind_path = os.getenv('ACTIVE_ENV', 'staging2') + "/minds/" + mind_id + "/mind.pkl"
    mind_dl_path = os.path.join(os.sep, 'tmp', 'mind.pkl')
    s3.Bucket(bucket).download_file(mind_path,mind_dl_path)
    mind_dict = pickle.load(open(mind_dl_path,'rb'))
    return mind_dict
# -


def handler(event, context):
    if isinstance(event['body'], str):
        json_request = json.loads(event['body'])
    else:
        json_request = event['body']
    #logger.info("POST request recieved", extra={"request": json_request})
    Request_obj = decode_json_request(json_request)
    mindId = str(json_request['mindId']).lower()
    #mind_dict = load_mind_features(mindId)
    lambda_function = "mind-" + mindId
    if not Request_obj.segments:
        return json({"msg": "No segments to process"})
    topics = {}
    pim = {}
    topics, pim = get_groups(Request_obj, lambda_function)

    topics['contextId'] = (json_request)['contextId']
    topics['instanceId'] = (json_request)['instanceId']
    topics['mindId'] = mindId
    output_pims = format_pims_output(pim, json_request, Request_obj.segments_map, mindId)
    return output_pims
