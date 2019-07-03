from text_preprocessing import preprocess as tp
import jgtextrank as tr
import networkx as nx
import matplotlib.pyplot as plt
# from sanic import Sanic
import pandas as pd
from datetime import datetime
import iso8601
import time
import json as js
from operator import itemgetter


def formatTime(tz_time):
    isoTime = iso8601.parse_date(tz_time)
    ts = isoTime.timestamp()
    ts = datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S:%f")
    return ts


def reformat_input(input_json):
    systime = time.strftime('%d-%m-%Y-%H-%M')
    json_path = 'call_summary_d2v_' + systime + '.json'

    json_df_ts = pd.DataFrame(input_json['segments'], index=None)
    json_df_ts['id'] = json_df_ts['id'].astype(str)
    json_df_ts['filteredText'] = json_df_ts['filteredText'].apply(lambda x: str(x))
    json_df_ts['originalText'] = json_df_ts['originalText'].apply(lambda x: str(x))
    json_df_ts['createdAt'] = json_df_ts['createdAt'].apply(lambda x: formatTime(x))
    json_df_ts['endTime'] = json_df_ts['endTime'].apply(lambda x: formatTime(x))
    json_df_ts['startTime'] = json_df_ts['startTime'].apply(lambda x: formatTime(x))
    json_df_ts['updatedAt'] = json_df_ts['updatedAt'].apply(lambda x: formatTime(x))
    json_df_ts = json_df_ts.sort_values(['createdAt'], ascending=[1])

    return json_df_ts


def read_segments(segment_df):
    for i in range(len(segment_df)):
        segment_text = segment_df.iloc[i]['originalText']

        yield segment_text


def write_segments_to_file(segment_df, file_name):
    for segments in read_segments(segment_df):
        with open(file_name, 'a') as f_:
            f_.write(segments)

    return file_name


def read_doc(doc_path):
    """

    Args:
        doc_path:

    Returns:
        full_text:
    """

    full_text = ''
    try:
        with open(doc_path, 'r') as doc:
            full_text = doc.read()

    except FileNotFoundError:
        print("{} file does not exist".format(doc_path))

    return full_text


def preprocess_text(text, stop_words=True, word_tokenize=True, remove_punct=False):
    """
    Preprocessing handles:
        - Remove punctuations
        - Expand contractions
        - Remove irrelevant symbols
        - Change dates
        - Change numbers

    Args:
        remove_punct:
        word_tokenize:
        stop_words:
        text:

    Returns:
        filtered_segmented_text
    """
    filtered_segmented_text = tp.preprocess(text, stop_words=stop_words, word_tokenize=word_tokenize,
                                            remove_punct=remove_punct)

    return filtered_segmented_text


def get_keyphrases_from_segmented_text(filtered_segmented_input, top_n='all', custom_stop_words=None):
    """

    Args:
        top_n:
        custom_stop_words:
        filtered_segmented_text:

    Returns:
        keyphrases_list:
    """

    if top_n == 'all':
        keyphrases_list = \
        tr.keywords_extraction_from_segmented_corpus(filtered_segmented_input, conn_with_original_ctx=False,
                                                     weight_comb='log_norm_max',
                                                     window=4,
                                                     top_p=1,
                                                     stop_words=custom_stop_words)[0]
    else:
        keyphrases_list = \
        tr.keywords_extraction_from_segmented_corpus(filtered_segmented_input, conn_with_original_ctx=False,
                                                     weight_comb='log_norm_max',
                                                     window=4,
                                                     top_p=1,
                                                     stop_words=custom_stop_words)[0][:top_n]

    return keyphrases_list


def get_keyphrases_from_text(input_text, top_n='all', custom_stop_words=None):
    if top_n == 'all':
        keyphrases_list = \
        tr.keywords_extraction(input_text, conn_with_original_ctx=False,
                                                     weight_comb='log_norm_max',
                                                     window=4,
                                                     top_p=1,
                                                     stop_words=custom_stop_words)[0]
    else:
        keyphrases_list = \
        tr.keywords_extraction(input_text, conn_with_original_ctx=False,
                                                     weight_comb='log_norm_max',
                                                     window=4,
                                                     top_p=1,
                                                     stop_words=custom_stop_words)[0][:top_n]

    return keyphrases_list


def segment_search(input_json, keyphrase_list):
    """
    Search for keyphrases in the top-5 PIM segments and return them as final result
    Args:
        input_segment:
        keyphrase_list:

    Returns:

    """
    input_segment = input_json['segments'][0].get('originalText').lower()
    keywords_list = []
    for tup in keyphrase_list:
        kw = tup[0]
        score = tup[1]

        result = input_segment.find(kw)
        if result > 0:
            keywords_list.append((kw, score))

    sort_list = sorted(keywords_list, key=itemgetter(1), reverse=True)
    return sort_list

def build_cooccur_graph(input_text, window=2, syntactic_filter=None):
    proc_input_text = tr.preprocessing(input_text)
    co_graph, tokens = tr.build_cooccurrence_graph(proc_input_text, window=window)
    
    return co_graph, tokens, proc_input_text


def main():
    json_file_path = "sample_req.json"
    with open(json_file_path) as f_:
        json_req = js.load(f_)

    segment_df = reformat_input(json_req)
    full_text_file = write_segments_to_file(segment_df=segment_df, file_name='input_text.txt')

    full_text = read_doc(doc_path=full_text_file)
    # filtered_text = preprocess_text(text=full_text)

    custom_stop_words = ['i', 'we', 'yeah', 'okay', 'like', 'mean', 'think', 'things', 'right', 'thing']
    keyphrase_list = get_keyphrases_from_text(full_text, custom_stop_words=custom_stop_words)

    print(keyphrase_list)
    print('\n')

    inp_req = 'inp_req.json'
    with open(inp_req) as inp:
        inp_json = js.load(inp)

    segment_keyword_list = segment_search(input_json=inp_json, keyphrase_list=keyphrase_list)
    print(segment_keyword_list)

    print('\n')

    # Testing on a single segment

    # input_segment = inp_json['segments'][0].get('originalText')
    # kw = get_keyphrases_from_text(input_segment, custom_stop_words=custom_stop_words)
    # print(kw)


if __name__ == '__main__':
    main()
