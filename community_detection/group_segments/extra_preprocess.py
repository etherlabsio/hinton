# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: sri_gpt
#     language: python3
#     name: sri_gpt
# ---

import sys
sys.path.append("../../../ai-engine_temp/pkg/")
import text_preprocessing.preprocess as tp
import nltk
import iso8601
from datetime import datetime
from copy import deepcopy
import json
import string
import itertools


# +


from nltk.corpus import stopwords
stop_words = stopwords.words("english")
def st_get_candidate_phrases(text, pos_search_pattern_list=[r"""base: {<(CD)|(DT)|(JJR)>* (<VB.>*)( (<NN>+ <NN>+)|((<JJ>|<NN>) <NN>)| ((<JJ>|<NN>)+|((<JJ>|<NN>)* (<NN> <NN.>)? (<JJ>|<NN>)*) <NN.>))}"""]
):
        punct = set(string.punctuation)
        all_chunks = []

        for pattern in pos_search_pattern_list:
            all_chunks+=st_getregexChunks(text,pattern)

        candidates_tokens = [' '.join(word for word, pos, 
                    chunk in group).lower() 
                    for key, group in itertools.groupby(all_chunks, 
                    lambda_unpack(lambda word, pos, chunk: chunk != 'O')) if key]

        candidate_phrases = [cand for cand in candidates_tokens if cand not in stop_words and not all(char in punct for char in cand)]
        #print (candidate_phrases)
        return candidate_phrases
    
def st_getregexChunks(text,grammar):

    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    #print(grammar)
    #print(all_chunks)
    #print()

    return all_chunks

def lambda_unpack(f):
    return lambda args: f(*args)

def get_filtered_pos(filtered, pos_list=['NN', 'JJ']):
    filtered_list_temp = []
    filtered_list = []
    flag = False
    flag_JJ = False
    for word, pos in filtered:
        if pos == 'NN' or pos == 'JJ':
            flag=True
            if pos == 'JJ':
                flag_JJ = True
            else:
                flag_JJ = False
            filtered_list_temp.append((word, pos))
            continue
        if flag:
            if 'NN' in list(map(lambda x: x[1], filtered_list_temp)):
                if not flag_JJ:
                    filtered_list.append(list(map(lambda x:x[0], filtered_list_temp)))
                else:
                    filtered_list.append(list(map(lambda x:x[0], filtered_list_temp))[:-1])
                    #print (filtered_list_temp)
                    #print (filtered_list[-1])
                    flag_JJ = False
            filtered_list_temp = []
            flag=False
            
    return filtered_list
# -

def preprocess_text(text):
    mod_texts_unfiltered = tp.preprocess(text, stop_words=False, remove_punct=False)
    mod_texts = []
    if mod_texts_unfiltered is not None:
        for index, sent in enumerate(mod_texts_unfiltered):

            #pos_tagged_sent = tp.preprocess(sent, stop_words=False, pos=True)[1][0]
            #filtered_list = get_filtered_pos(pos_tagged_sent)
            filtered_list = st_get_candidate_phrases(sent)
            if len(filtered_list)==0:
                continue
            elif True not in list(map(lambda x: len(x.split(' '))>=1, filtered_list)):
                if len(filtered_list) > 2:
                    pass
                else:
                    continue
                #continue

            if len(sent.split(' ')) > 250:
                length = len(sent.split(' '))
                split1 = ' '.join([i for i in sent.split(' ')[:round(length / 2)]])
                split2 = ' '.join([i for i in sent.split(' ')[round(length / 2):]])
                mod_texts.append(split1)
                mod_texts.append(split2)
                continue

            if len(sent.split(' ')) <= 10:
                    continue

            mod_texts.append(sent)
        if len(mod_texts) ==1:
            if not (len(mod_texts[0].split(' ')) >= 20):
                return ""
        #elif len(mod_texts) == 0:
        #    return ""
        
    else:
        return ""
    
    if mod_texts == []:
        return ""
    return mod_texts


t_list = preprocess_text("It looks like a regression because it did not happen before and also it looks like when in the in a similar scenario when you are on iOS you do not run into that issue. Also need to resubmit the slack and iOS apps we are going to call the meter Labs instead of ether meet because it now has meat costs and notes. There is a bunch of changes related to that we need to do if especially the iOS team if you can take a look at the app and get it ready for the submission.")

[st_get_candidate_phrases(x) for x in t_list]

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
            new_seg = {}
            new_seg = deepcopy(segmentsmap[pim[no][seg][-1]])
            # new_seg["originalText"] = pim[no][seg][0]
            tmp_seg.append(new_seg)
        pims["group"][no] = tmp_seg
    # pims["group"] = pim
    pims['contextId'] = (req)['contextId']
    pims['instanceId'] = (req)['instanceId']
    pims['mindId'] = mindId
    response_output = {}
    response_output['statusCode'] = 200
    response_output['headers'] = {"Content-Type": "application/json"}
    response_output['body'] = json.dumps(pims)
    return response_output
