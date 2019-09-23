import sys
sys.path.append("../../../ai-engine/pkg/")
import text_preprocessing.preprocess as tp
import nltk
import iso8601
from datetime import datetime
import json


def preprocess_text(text):
    mod_texts_unfiltered = tp.preprocess(text, stop_words=False, remove_punct=False)
    mod_texts = []

    for index, sent in enumerate(mod_texts_unfiltered):
        if len(sent.split(' ')) > 250:
            length = len(sent.split(' '))
            split1 = ' '.join([i for i in sent.split(' ')[:round(length / 2)]])
            split2 = ' '.join([i for i in sent.split(' ')[round(length / 2):]])
            mod_texts.append(split1)
            mod_texts.append(split2)
            continue

        if len(sent.split(' ')) <= 6:
            continue

        mod_texts.append(sent)
    return mod_texts


def format_time(tz_time, datetime_object=False):
    isoTime = iso8601.parse_date(tz_time)
    ts = isoTime.timestamp()
    ts = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S:%f")

    if datetime_object:
        ts = datetime.fromisoformat(ts)
    return ts


def format_pims_output(pim, req, segmentsmap, mindId):
    pims = {}
    pims["group"] = {}
    for no in pim.keys():
        tmp_seg = []
        for seg in pim[no].keys():
            tmp_seg.append(segmentsmap[pim[no][seg][-1]])
        pims["group"][no] = tmp_seg
    pims['contextId'] = (req)['contextId']
    pims['instanceId'] = (req)['instanceId']
    pims['mindId'] = mindId
    response_output = {}
    response_output['statusCode'] = 200
    response_output['headers'] = {"Content-Type": "application/json"}
    response_output['body'] = json.dumps(pims)
    return response_output
