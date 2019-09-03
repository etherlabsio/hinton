try:
    import unzip_requirements
except ImportError:
    pass

import os
import io
import json
import glob
import time
import logging

import boto3
import requests

import torch
from gpt import OpenAIGPTPreTrainedModel,OpenAIGPTConfig,OpenAIGPTModel,SequenceSummary
from mind_utils import getSentenceFeatures, getParagraphFeatures, CustomOpenAIGPTDoubleHeadsModel, NumpyEncoder
import numpy as np
import pickle

s3 = boto3.resource('s3')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def load_model_files():

    bucket = os.getenv('BUCKET_NAME', 'io.etherlabs.gpt.artifacts')
    model_path = os.getenv('MODEL')
    mind_path = os.getenv('MIND')
    config_path = os.getenv('CONFIG')
    modelObj = s3.Object(
        bucket_name=bucket,
        key=model_path
    )
    mind_dl_path = os.path.join(os.sep, 'tmp', 'mind.pkl')
    s3.Bucket(bucket).download_file(mind_path,mind_dl_path)

    config_dl_path = os.path.join(os.sep, 'tmp', 'gpt_config.json')
    s3.Bucket(bucket).download_file(config_path,config_dl_path)

    state_dict = torch.load(io.BytesIO(modelObj.get()["Body"].read()),map_location='cpu')
    mind_dict = pickle.load(open(mind_dl_path,'rb'))
    return state_dict,mind_dict,config_dl_path

def process_input(json_request):
    if isinstance(json_request, str):
        json_request = json.loads(json_request)
    text_request = json_request['text']
    return text_request

# load the model when lambda execution context is created
state_dict,mind_dict, config_path = load_model_files()
config = OpenAIGPTConfig(config_path)
config.vocab_size = config.vocab_size + config.n_special
model = CustomOpenAIGPTDoubleHeadsModel(config)
model.load_state_dict(state_dict)
model.eval()

def lambda_handler(event, context):
    logger.info(event)
    input_text = process_input(event['body'])
    model_feats = getSentenceFeatures(model, mind_dict, input_text)
    response = json.dumps(model_feats,cls=NumpyEncoder)

    return {
        "statusCode": 200,
        "body" : response
    }
