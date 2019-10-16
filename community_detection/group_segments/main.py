import json
import sys
sys.path.append("../../../ai-engine/pkg")
from transport import decode_json_request
from grouper import get_groups
from extra_preprocess import format_pims_output
import sys
import logging
from log.logger import setup_server_logger

logger = logging.getLogger()
setup_server_logger(debug=False)


def handler(event, context):
    if isinstance(event['body'], str):
        json_request = json.loads(event['body'])
    else:
        json_request = event['body']
    #logger.info("POST request recieved", extra={"request": json_request})
    Request_obj = decode_json_request(json_request)
    mindId = str(json_request['mindId']).lower()
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
