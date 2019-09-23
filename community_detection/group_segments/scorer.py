from numpy import dot
from numpy.linalg import norm
from boto3 import client as boto3_client
import json
import logging
from botocore.client import Config
import numpy as np

logger = logging.getLogger(__name__)

config = Config(connect_timeout=240, read_timeout=240, retries={'max_attempts': 0}, )
lambda_client = boto3_client('lambda', config=config)


def cosine(vec1, vec2):
    return dot(vec1, vec2) / (norm(vec1) * norm(vec2))


def get_feature_vector(mind_input, lambda_function):
    # logger.info("computing feature vector", extra={"msg": "getting feature vector from mind service"})
    logger.info("getting feature vector from mind service")
    invoke_response = lambda_client.invoke(FunctionName=lambda_function, InvocationType='RequestResponse', Payload=mind_input)
    #invoke_response = lambda_client.invoke(FunctionName="keyphrase_ranker", InvocationType='RequestResponse', Payload=mind_input)
    logger.info("Request Sent")
    # logger.info("computing feature vector", extra={"msg": "Request Sent"})
    out_json = invoke_response['Payload'].read().decode('utf8').replace("'", '"')
    data = json.loads(json.loads(out_json)['body'])
    response = json.loads(out_json)['statusCode']

    if response == 200:
        feature_vector = data['sent_feats'][0]
        logger.info("Response Recieved")

        # logger.info("computing feature vector", extra={"msg": "Response Recieved"})
    else:
        logger.error("Invalid response from  mind service")
        print(response)
        # logger.error("computing feature vector", extra={"msg": "Invalid response from  mind service"})
    return feature_vector


def get_embeddings(input_list, req_data=None):
    aws_config = Config(
        connect_timeout=60,
        read_timeout=300,
        retries={"max_attempts": 0},
        region_name="us-east-1",
    )
    lambda_client = boto3_client("lambda", config=aws_config)
    if req_data is None:

        lambda_payload = {"body": {"text_input": input_list}}
    else:
        lambda_payload = {"body": {"request": req_data, "text_input": input_list}}

    try:
        #logger.info("Invoking lambda function")
        invoke_response = lambda_client.invoke(
            FunctionName="keyphrase_ranker",
            InvocationType="RequestResponse",
            Payload=json.dumps(lambda_payload),
        )

        lambda_output = (
            invoke_response["Payload"].read().decode("utf8").replace("'", '"')
        )
        response = json.loads(lambda_output)
        status_code = response["statusCode"]
        response_body = response["body"]

        if status_code == 200:
            embedding_vector = np.asarray(json.loads(response_body)["embeddings"])

        else:
            embedding_vector = np.asarray(json.loads(response_body)["embeddings"])
    except Exception as e:
        print (e)
        pass
    return embedding_vector
