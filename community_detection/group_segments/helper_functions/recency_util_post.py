import sys
sys.path.append("/home/ray__/ssd/pos_tag/code/") # Custom pos tag location.
sys.path.append("/home/ray__/ssd/") # ner util code
import recency_util_post

from bert_ner_utils_graph import BERT_NER
from distilbert_pos_tagger import DistilBertPosTagger

pos_tagger = DistilBertPosTagger("/home/ray__/ssd/pos_tag/model/Distilbert/","cpu")
ner_model = BERT_NER('/home/ray__/ssd/bert-ner/',labels = ["O", "MISC", "MISC",  "PER", "PER", "ORG", "ORG", "LOC", "LOC"],device="cpu")

sys.path.append("/home/ray__/ssd/BERT/") # gpt model utils location
from gpt_feat_utils import GPT_Inference
gpt_model = GPT_Inference("/home/ray__/ssd/BERT/models/se/epoch3/", device="cpu")

import pickle
import pandas as pd
import numpy as np
import os
from io import open
import re
import glob
from nltk.tokenize import sent_tokenize
import json
import nltk, string,itertools
from collections import Counter
import networkx as nx
from nltk.corpus import stopwords
import sys

WEB_URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""

class CandidateKPExtractor(object):

    def __init__(self, stop_words, filter_small_sents = True):

        self.punct = set(string.punctuation)
        self.filter_small_sents = filter_small_sents
        self.stop_words = stop_words

    def get_candidate_phrases(self, text, pos_search_pattern_list=[r"""base: {(<JJ.*>*<NN.*>+<IN>)?<JJ>*<NN.*>+}"""]):
                                        #r"""nounverb:{<NN.+>+<.+>{0,2}<VB+>{1}}""",
                                        #r"""verbnoun:{<VB.+>{1}<.+>{0,5}<NN.+>+}"""]):
        all_chunks = []

        for pattern in pos_search_pattern_list:
            all_chunks+=self.getregexChunks(text, pattern)

        candidates_tokens = [' '.join(word for word, pos, 
                    chunk in group).lower()
                    for key, group in itertools.groupby(all_chunks, 
                    self.lambda_unpack(lambda word, pos, chunk: chunk != 'O')) if key]

        candidate_phrases = [cand for cand in candidates_tokens if cand not in self.stop_words and not all(char in self.punct for char in cand)]

        return candidate_phrases

    def getregexChunks(self,text, grammar):

        chunker = nltk.chunk.regexp.RegexpParser(grammar)
#         tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
        tagged_sents = [pos_tagger.pos_tag(sent) for sent in nltk.sent_tokenize(text)]
        all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                        for tagged_sent in tagged_sents))
        return all_chunks

    def lambda_unpack(self, f):
        return lambda args: f(*args)

    
def cleanText(text,stripSpecial=False):
    
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF""]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = replaceContractions(text)
    text = text.replace('\\n','').replace("’","'").replace('\\','').replace("‘","'")
    #text = re.sub('(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?\s',
    #            '',text)
    text = re.sub(WEB_URL_REGEX,'__url__', text)
    text = text.replace(':','. ').replace("”","'").replace("“","'").replace('̶',"")
    text = text.replace("\u200a—\u200a",' ').replace('\xa0','')
    if stripSpecial:
        text = re.sub('\W', ' ', text)
    text = re.sub(' +', ' ', text)
    text = text.replace('\t', '')
    text = text.replace('\n', '')
    return text.strip()

def replaceContractions(text):
    #text = text.lower()
    c_filt_text = ''
    for word in text.split(' '):
        if word in list(contractions['contractions']):
            c_filt_text = c_filt_text+' '+contractions['contractions'][word]
        else:
            c_filt_text = c_filt_text+' '+word
    return c_filt_text

def simpleFilter(text):
    return re.sub(r'\W+', ' ', text)

def splitText(text):
    # returns list of sentences
    text = text.strip()
    if not text.endswith((".","?","!")):
        text+="."
    text = text.replace("?.","?")
    split_text = nltk.sent_tokenize(text)
    return split_text

def paraSplit(para):
    para = cleanText(para)
    tok_sents = [ele for ele in splitText(para) if len(nltk.word_tokenize(ele))>5]
    merged_text = ' '.join(tok_sents)
    return merged_text
def articleSplit(article, remove_extra=False):
#     article = article.encode('ascii', 'ignore').decode('ascii')
    if remove_extra:
        article = article[2:]
    paras = re.sub("[.][A-Z]",lambda x: x.group(0)[0]+"\n"+x.group(0)[1],article).split("\n")
    return [paraSplit(para) for para in paras]


contractions = {'contractions': {"ain't": 'am not', "aren't": 'are not', "can't": 'can not', "can't've": 'can not have', "'cause": 'because', "could've": 'could have', "couldn't": 'could not', "couldn't've": 'could not have', "didn't": 'did not', "doesn't": 'does not', "don't": 'do not', "hadn't": 'had not', "hadn't've": 'had not have', "hasn't": 'has not', "haven't": 'have not', "he'd": 'he would', "he'd've": 'he would have', "he'll": 'he will', "he's": 'he is', "how'd": 'how did', "how'll": 'how will', "how's": 'how is', "i'd": 'i would', "i'll": 'i will', "i'm": 'i am', "i've": 'i have', "isn't": 'is not', "it'd": 'it would', "it'll": 'it will', "it's": 'it is', "let's": 'let us', "ma'am": 'madam', "mayn't": 'may not', "might've": 'might have', "mightn't": 'might not', "must've": 'must have', "mustn't": 'must not', "needn't": 'need not', "oughtn't": 'ought not', "shan't": 'shall not', "sha'n't": 'shall not', "she'd": 'she would', "she'll": 'she will', "she's": 'she is', "should've": 'should have', "shouldn't": 'should not', "that'd": 'that would', "that's": 'that is', "there'd": 'there had', "there's": 'there is', "they'd": 'they would', "they'll": 'they will', "they're": 'they are', "they've": 'they have', "wasn't": 'was not', "we'd": 'we would', "we'll": 'we will', "we're": 'we are', "we've": 'we have', "weren't": 'were not', "what'll": 'what will', "what're": 'what are', "what's": 'what is', "what've": 'what have', "where'd": 'where did', "where's": 'where is', "who'll": 'who will', "who's": 'who is', "won't": 'will not', "wouldn't": 'would not', "you'd": 'you would', "you'll": 'you will', "you're": 'you are', "i'll": 'i will', "we'll": 'we will', "let's": 'let us', "i'd": 'i would'}}
stop_words = {'a', 'able', 'about', 'above', 'abst', 'accordance', 'according', 'accordingly', 'across', 'act', 'actually', 'added', 'adj', 'affected', 'affecting', 'affects', 'after', 'afterwards', 'again', 'against', 'ah', 'all', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'alright', 'am', 'among', 'amongst', 'an', 'and', 'announce', 'another', 'any', 'anybody', 'anyhow', 'anymore', 'anyone', 'anything', 'anyway', 'anyways', 'anywhere', 'apparently', 'approximately', 'are', 'aren', 'arent', 'arise', 'around', 'as', 'aside', 'ask', 'asking', 'at', 'auth', 'available', 'away', 'awfully', 'b', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming', 'been', 'before', 'beforehand', 'begin', 'beginning', 'beginnings', 'begins', 'behind', 'being', 'believe', 'below', 'beside', 'besides', 'between', 'beyond', 'biol', 'both', 'brief', 'briefly', 'but', 'by', 'c', 'ca', 'came', 'can', 'cannot', "can't", 'cause', 'causes', 'certain', 'certainly', 'co', 'com', 'come', 'comes', 'contain', 'containing', 'corresponding', 'contains', 'could', 'couldnt', 'd', 'date', 'did', "didn't", 'different', 'do', 'does', "doesn't", 'doing', 'done', "don't", 'dont', 'down', 'downwards', 'due', 'during', 'e', 'each', 'ed', 'edu', 'effect', 'eg', 'eight', 'eighty', 'either', 'else', 'elsewhere', 'end', 'ending', 'enough', 'especially', 'et', 'et-al', 'etc', 'even', 'ever', 'every', 'everybody', 'everyone', 'everything', 'everywhere', 'ex', 'except', 'f', 'far', 'few', 'ff', 'fifth', 'first', 'five', 'fix', 'followed', 'following', 'follows', 'for', 'former', 'formerly', 'forth', 'found', 'four', 'from', 'further', 'furthermore', 'g', 'gave', 'get', 'gets', 'getting', 'give', 'given', 'gives', 'giving', 'go', 'goes', 'gone', 'got', 'gotten', 'h', 'had', 'happens', 'hardly', 'has', "hasn't", 'have', "haven't", 'having', 'he', 'hed', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'heres', 'hereupon', 'hers', 'herself', 'hes', 'hi', 'hid', 'him', 'himself', 'his', 'hither', 'home', 'how', 'howbeit', 'however', 'hundred', 'i', 'id', 'ie', 'if', "i'll", 'im', 'immediate', 'immediately', 'importance', 'important', 'in', 'inc', 'indeed', 'index', 'information', 'instead', 'into', 'invention', 'inward', 'is', "isn't", 'it', 'itd', "it'll", 'its', 'itself', "i've", 'j', 'just', 'k', 'keep', 'keeps', 'kept', 'kg', 'km', 'know', 'known', 'knows', 'l', 'largely', 'last', 'lately', 'later', 'latter', 'latterly', 'least', 'less', 'lest', 'let', 'lets', 'like', 'liked', 'likely', 'line', 'little', "'ll", 'look', 'looking', 'looks', 'ltd', 'm', 'made', 'mainly', 'make', 'makes', 'many', 'may', 'maybe', 'me', 'mean', 'means', 'meantime', 'meanwhile', 'merely', 'mg', 'might', 'million', 'miss', 'ml', 'more', 'moreover', 'most', 'mostly', 'mr', 'mrs', 'much', 'mug', 'must', 'my', 'myself', 'n', 'na', 'name', 'namely', 'nay', 'nd', 'near', 'nearly', 'necessarily', 'necessary', 'need', 'needs', 'neither', 'never', 'nevertheless', 'new', 'next', 'nine', 'ninety', 'no', 'nobody', 'non', 'none', 'nonetheless', 'noone', 'nor', 'normally', 'nos', 'not', 'noted', 'nothing', 'now', 'nowhere', 'o', 'obtain', 'obtained', 'obviously', 'of', 'off', 'often', 'oh', 'ok', 'okay', 'old', 'omitted', 'on', 'once', 'one', 'ones', 'only', 'onto', 'or', 'ord', 'other', 'others', 'otherwise', 'ought', 'our', 'ours', 'ourselves', 'out', 'outside', 'over', 'overall', 'owing', 'own', 'p', 'page', 'pages', 'part', 'particular', 'particularly', 'past', 'per', 'perhaps', 'placed', 'please', 'plus', 'poorly', 'possible', 'possibly', 'potentially', 'pp', 'predominantly', 'present', 'previously', 'primarily', 'probably', 'promptly', 'proud', 'provides', 'put', 'q', 'que', 'quickly', 'quite', 'qv', 'r', 'ran', 'rather', 'rd', 're', 'readily', 'really', 'recent', 'recently', 'ref', 'refs', 'regarding', 'regardless', 'regards', 'related', 'relatively', 'research', 'respectively', 'resulted', 'resulting', 'results', 'right', 'run', 's', 'said', 'same', 'saw', 'say', 'saying', 'says', 'sec', 'section', 'see', 'seeing', 'seem', 'seemed', 'seeming', 'seems', 'seen', 'self', 'selves', 'sent', 'seven', 'several', 'shall', 'she', 'shed', "she'll", 'shes', 'should', "shouldn't", 'show', 'showed', 'shown', 'showns', 'shows', 'significant', 'significantly', 'similar', 'similarly', 'since', 'six', 'slightly', 'so', 'some', 'somebody', 'somehow', 'someone', 'somethan', 'something', 'sometime', 'sometimes', 'somewhat', 'somewhere', 'soon', 'sorry', 'specifically', 'specified', 'specify', 'specifying', 'still', 'stop', 'strongly', 'sub', 'substantially', 'successfully', 'such', 'sufficiently', 'suggest', 'sup', 'sure\tt', 'take', 'taken', 'taking', 'tell', 'tends', 'th', 'than', 'thank', 'thanks', 'thanx', 'that', "that'll", 'thats', "that've", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'thered', 'therefore', 'therein', "there'll", 'thereof', 'therere', 'theres', 'thereto', 'thereupon', "there've", 'these', 'they', 'theyd', "they'll", 'theyre', "they've", 'think', 'thing', 'things', 'this', 'those', 'thou', 'though', 'thoughh', 'thousand', 'throug', 'through', 'throughout', 'thru', 'thus', 'til', 'tip', 'to', 'together', 'too', 'took', 'toward', 'towards', 'tried', 'tries', 'truly', 'try', 'trying', 'ts', 'twice', 'two', 'u', 'un', 'under', 'unfortunately', 'unless', 'unlike', 'unlikely', 'until', 'unto', 'up', 'upon', 'ups', 'us', 'use', 'used', 'useful', 'usefully', 'usefulness', 'uses', 'using', 'usually', 'v', 'value', 'various', "'ve", 'very', 'via', 'viz', 'vol', 'vols', 'vs', 'w', 'want', 'wants', 'was', 'wasnt', 'way', 'we', 'wed', 'welcome', "we'll", 'went', 'were', 'werent', "we've", 'what', 'whatever', "what'll", 'whats', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'wheres', 'whereupon', 'wherever', 'whether', 'which', 'while', 'whim', 'whither', 'who', 'whod', 'whoever', 'whole', "who'll", 'whom', 'whomever', 'whos', 'whose', 'why', 'widely', 'willing', 'wish', 'with', 'within', 'without', 'wont', 'words', 'world', 'would', 'wouldnt', 'www', 'x', 'y', 'yes', 'yeah', 'yet', 'you', 'youd', "you'll", 'your', 'youre', 'yours', 'yourself', 'yourselves', "you've", 'z', 'zero', 'hear', 'see'}

kp_model = CandidateKPExtractor(stop_words)

def get_grouped_segments(groups):
    paragraphs = []
    meeting_groups = []
    for groupid, groupobj in groups.items():
        paragraphs.append(" ".join([seg["originalText"] for seg in groupobj]))
    meeting_groups.append(" <p_split>".join(paragraphs))
    
    ids = ["0"]
    return meeting_groups, ids

def form_sentence_graph(master_paragraphs, master_ids, multi_label_dict):
    ent_sent_dict = {}
    kp_sent_dict = {}
    label_dict = multi_label_dict
    noun_list = []
    for meeting_ctr in range(len(master_paragraphs)):
        clean_text= cleanText(master_paragraphs[meeting_ctr]).lower()
        tok_sents = nltk.sent_tokenize(clean_text)
        tok_sents = [ele for ele in tok_sents if len(nltk.word_tokenize(ele))>5]
        merged_text = ' '.join(tok_sents)

        paragraph_split = merged_text.split('<p_split>')
        paragraph_sets = []
        process_ctr = 0
        for i in range(len(paragraph_split)):
            if process_ctr>=len(paragraph_split):
                break
            curr_para = paragraph_split[process_ctr]
            if len(curr_para.split(' '))>5:
                if curr_para[-1]!='.':
                    curr_para = curr_para.strip()+'.'
                para_len = len(nltk.sent_tokenize(curr_para))
                if para_len>=4:
                    paragraph_sets.append(curr_para.replace('..','. ').replace(':.','.'))
                    process_ctr+=1
                else: 
                    while para_len<4 and process_ctr<len(paragraph_split)-1:
                        process_ctr+=1
                        text_to_add = paragraph_split[process_ctr].strip()
                        curr_para = curr_para+' '+text_to_add
                        if curr_para[-1]!='.':
                            curr_para = curr_para.strip()+'.'
                        para_len = len(nltk.sent_tokenize(curr_para))
                    curr_para = curr_para.replace('..','. ').replace(':.','.')
                    paragraph_sets.append(curr_para)
                    process_ctr+=1
            else:
                process_ctr+=1


        meet_id = master_ids[meeting_ctr]
        for p_no, para in enumerate(paragraph_sets):
            sent_list = sent_tokenize(para)
            master_sent_list = []
            master_entity_list = []
            master_kp_list = []

            for sent in sent_list:
                sent_kps = kp_model.get_candidate_phrases(sent)
                _, sent_ent_labels = ner_model.get_entities(sent,get_labels=True)
                if len(sent_ent_labels)>0:
                    master_sent_list.append(sent)
                    master_entity_list.append(sent_ent_labels)
                if len(sent_kps)>0:
                    master_kp_list.append(sent_kps)

            meet_id1 = str(meet_id)+"_"+str(p_no)
            for sent,ents in zip(master_sent_list,master_entity_list):
                for ent, lab in ents.items():
                    if ent in ent_sent_dict:
                        #update the value and assign
                        updated_sents = [sent]
                        if meet_id1 in ent_sent_dict[ent]:
                            curr_sents = list(ent_sent_dict[ent][meet_id1])
                            updated_sents = curr_sents+[sent]
                        ent_sent_dict[ent][meet_id1] = updated_sents
                        label_dict[ent].update([lab])
                    else:
                        ent_sent_dict[ent] = {meet_id1:[sent]}
                        label_counter = Counter(dict.fromkeys(ner_model.labels,0))
                        label_counter.update([lab])
                        label_dict[ent] = label_counter
            for sent,kps in zip(master_sent_list,master_kp_list):
                for kp in kps:
                    if kp in kp_sent_dict:
                        #update the value and assign
                        updated_sents = [sent]
                        if meet_id1 in kp_sent_dict[kp]:
                            curr_sents = list(kp_sent_dict[kp][meet_id1])
                            updated_sents = list(kp_sent_dict[kp][meet_id1])+[sent]
                        kp_sent_dict[kp][meet_id1] = updated_sents
                    else:
                        kp_sent_dict[kp] = {meet_id1:[sent]}
            single_nouns = list(set([ele for ele in kp_model.get_candidate_phrases(para,
                                                                [r"""singular_nn: {<NN>{1}}"""]) if len(ele.split(' '))==1]))
            single_nouns = [ele.lower() for ele in single_nouns]
            noun_list.extend(single_nouns)
    ent_single_label_dict = {node:max(counter,key=lambda x: counter[x]) for node,counter in label_dict.items()}
    entity_dict = {e:sum(dcts.values(),[]) for e,dcts in ent_sent_dict.items()}
    return ent_sent_dict, kp_sent_dict, label_dict, noun_list, entity_dict
# pickle.dump(ent_sent_dict,open('/home/ether/hdd/ether/graph_dumps/'+domain+'/'+domain+'_sent_dict_cont.pkl','wb'))
# pickle.dump(kp_sent_dict,open('/home/ether/hdd/ether/graph_dumps/'+domain+'/'+domain+'_kp_sent_dict_cont.pkl','wb'))
# pickle.dump(label_dict,open('/home/ether/hdd/ether/graph_dumps/'+domain+'/'+domain+'_label_dict_cont.pkl','wb'))


def form_single_label_dict(label_dict, ent_sent_dict, kp_sent_dict):
    ent_single_label_dict = {node:max(counter,key=lambda x: counter[x]) for node,counter in label_dict.items()}
    all_sent_dict = ent_sent_dict.copy()
    all_sent_dict.update(dict(map(lambda x: (x[0].lower(),x[1]),kp_sent_dict.items())))
    return ent_single_label_dict, all_sent_dict


def get_base_graph(artifacts_dir):
    graph_file_location = artifacts_dir + "kp_entity_graph.pkl"
    ent_kp_graph = pickle.load(open(graph_file_location,"rb"))
    if any([d.get("is_ether_node","missing")=="missing" for n,d in ent_kp_graph.nodes(data=True)]):
        nodes_list = list(ent_kp_graph.nodes())
        for i,node in enumerate(nodes_list):
            ent_kp_graph.nodes[node]['is_ether_node'] = False
            ent_kp_graph.nodes[node]['ether_meet_ctr'] = 0
            ent_kp_graph.nodes[node]['ether_grp_ctr'] = 0
            ent_kp_graph.nodes[node]['ether_sent_ctr'] = 0
            ent_kp_graph.nodes[node]['ether_meet_freq_list'] = 0
            for j,node1 in enumerate(nodes_list):
                if j>i:
                    if ent_kp_graph.has_edge(node,node1):
                        ent_kp_graph[node][node1]['ether_meet_ctr'] = 0
                        ent_kp_graph[node][node1]['ether_grp_ctr'] = 0
                        ent_kp_graph[node][node1]['ether_sent_ctr'] = 0
    return ent_kp_graph

def update_entity_nodes(ent_kp_graph, ent_sent_dict, multi_label_dict):
    node_list = []
    for ent in ent_sent_dict:
        if ent.isdigit():
            continue
        node_list.append(ent)
        meet_dict = dict()
        for p,s in ent_sent_dict[ent].items():
            meet_dict[p.split("_")[0]] = meet_dict.get(p.split("_")[0],[]) + s
        meet_freq = list(map(lambda sent_list: len(sent_list),meet_dict.values()))
        if ent in ent_kp_graph:
            ent_kp_graph.nodes()[ent]['is_ether_node'] = True
            ent_kp_graph.nodes()[ent]['node_type'] = "entity"
            #ent_kp_graph.nodes()[ent]['node_label'] = ent_single_label_dict.get(ent,"N/A")
            ent_kp_graph.nodes()[ent]['node_label'] = multi_label_dict.get(ent, "N/A")
            ent_kp_graph.nodes()[ent]['ether_grp_ctr'] = len(ent_sent_dict[ent])
            ent_kp_graph.nodes()[ent]['ether_meet_ctr'] = len(meet_dict)
            ent_kp_graph.nodes()[ent]['ether_sent_ctr'] = sum(meet_freq)
            ent_kp_graph.nodes()[ent]['ether_meet_freq_list'] = list(map(lambda sent_list: len(sent_list),ent_sent_dict[ent].values()))

        else:
            ent_kp_graph.add_node(ent,
                                  node_type = "entity",
                                  is_ether_node = True,
#                                   node_label = ent_single_label_dict.get(ent,"N/A"),
                                  node_label = multi_label_dict.get(ent, "N/A"),
                                  ether_grp_ctr = len(ent_sent_dict[ent]),
                                  ether_meet_ctr = len(meet_dict),
                                  ether_meet_freq_list = meet_freq,
                                  ether_sent_ctr = sum(meet_freq),
                                  art_ctr = 0,
                                  para_ctr = 0,
                                  sent_ctr = 0)
    return ent_kp_graph, node_list

def update_kp_nodes(ent_kp_graph, ent_sent_dict, node_list, kp_sent_dict):
    all_ent_sents = []
    x = [all_ent_sents.extend(sents)  for p_dict in ent_sent_dict.values() for sents in p_dict.values() ] 
    all_ent_sents = set(all_ent_sents)
    for kp in kp_sent_dict:
        node_list.append(kp.lower())
        meet_dict = dict()
        for p,s in kp_sent_dict[kp].items():
            meet_dict[p.split("_")[0]] = meet_dict.get(p.split("_")[0],[]) + s
        meet_freq = list(map(lambda sent_list: len(sent_list),meet_dict.values()))
        if kp.lower() in ent_kp_graph:
            ent_kp_graph.nodes()[kp.lower()]['is_ether_node'] = True
            ent_kp_graph.nodes()[kp.lower()]['node_type'] = "key_phrase"
            ent_kp_graph.nodes()[kp.lower()]['node_label'] = "N/A"
            ent_kp_graph.nodes()[kp.lower()]['ether_grp_ctr'] = len(kp_sent_dict[kp])
            ent_kp_graph.nodes()[kp.lower()]['ether_meet_ctr'] = len(meet_dict)
            ent_kp_graph.nodes()[kp.lower()]['ether_meet_freq_list'] = meet_freq
            ent_kp_graph.nodes()[kp.lower()]['ether_sent_ctr'] = sum(meet_freq)
        else:
            ent_kp_graph.add_node(kp.lower(),
                                  node_type = "key_phrase",
                                  node_label = "N/A",
                                  is_ether_node = True,
                                  ether_grp_ctr = len(kp_sent_dict[kp]),
                                  ether_meet_ctr = len(meet_dict),
                                  ether_meet_freq_list = meet_freq,
                                  ether_sent_ctr = sum(meet_freq),
                                  art_ctr = 0,
                                  para_ctr = 0,
                                  sent_ctr = 0)
    return ent_kp_graph, node_list

def update_edges(ent_kp_graph, node_list, all_sent_dict, artifacts_dir):
    graph_file = artifacts_dir + "kp_entity_graph.pkl"
    node_type_map = {"entity":"ent","key_phrase":"kp"}
    for a, node_a in enumerate(node_list):
        for b, node_b in enumerate(node_list):
            if b>a:
                node_type_a = ent_kp_graph.nodes()[node_a]['node_type']
                node_type_b = ent_kp_graph.nodes()[node_b]['node_type']
                node_typestring_a = node_type_map[node_type_a]
                node_typestring_b = node_type_map[node_type_b]
                grp_set_a = set(all_sent_dict[node_a])
                grp_set_b = set(all_sent_dict[node_b])
                grp_intersection = grp_set_a & grp_set_b
                if len(grp_intersection)<1:
                    continue
                meet_set_a = set(list(map(lambda x: x.split("_")[0],grp_set_a)))
                meet_set_b = set(list(map(lambda x: x.split("_")[0],grp_set_b)))
                meet_intersection = meet_set_a & meet_set_b
                sent_set_a = set(list(itertools.chain(*list(all_sent_dict[node_a].values()))))
                sent_set_b = set(list(itertools.chain(*list(all_sent_dict[node_b].values()))))
                sent_intersection = sent_set_a & sent_set_b
                if node_typestring_a=="kp" and len(sent_intersection)<2:
                    continue
                if node_typestring_b=="kp" and len(sent_intersection)<1:
                    continue
                if ent_kp_graph.has_edge(node_a,node_b):
                    ent_kp_graph[node_a][node_b]['ether_meet_ctr'] = len(meet_intersection)
                    ent_kp_graph[node_a][node_b]['ether_grp_ctr'] = len(grp_intersection)
                    ent_kp_graph[node_a][node_b]['ether_sent_ctr'] = len(sent_intersection)
                ent_kp_graph.add_edge(node_a,
                                      node_b,
                                      edge_type = node_typestring_a + "_to_" + node_typestring_b,
                                      ether_meet_ctr = len(meet_intersection),
                                      ether_grp_ctr = len(grp_intersection),
                                      ether_sent_ctr = len(sent_intersection),
                                      art_ctr = 0,
                                      para_ctr = 0,
                                      sent_ctr = 0)
        if a%100==0:
            pickle.dump(ent_kp_graph,open(graph_file,"wb"))
            ent_kp_graph = pickle.load(open(graph_file,"rb"))
    pickle.dump(ent_kp_graph,open(graph_file,"wb"))
    return ent_kp_graph

def update_kp_tokens(ent_kp_graph, noun_list):
    entity_list = [node for node, d in ent_kp_graph.nodes(data=True) if d['node_type']=='entity']
    entity_list_lower = entity_list_lower = [ele.lower() for ele in entity_list]
    kp_nodes = [node for node, d in ent_kp_graph.nodes(data=True) if d['node_type']=='key_phrase']
    multi_tok_kps = [ele for ele in kp_nodes if len(ele.split(' '))>1]
    single_tok_kps = [ele for ele in kp_nodes if len(ele.split(' '))==1]
    multi_tok_kps = list(set(multi_tok_kps)-set(entity_list_lower))
    
    multi_kp_tokens = []
    for kp in multi_tok_kps:
        multi_kp_tokens.extend(kp.split(' '))
    multi_kp_tokens = list(set(multi_kp_tokens) - set(single_tok_kps) - set(entity_list_lower))
    noun_graph_tokens = list(set(multi_kp_tokens)&set(noun_list))
    nouns_to_update = []
    for noun_token in noun_graph_tokens:
        if noun_token in ent_kp_graph:
            ent_kp_graph.nodes()[noun_token]['is_ether_node'] = True
            ent_kp_graph.nodes()[noun_token]['node_label'] = "N/A"
            ent_kp_graph.nodes()[noun_token]['ether_meet_ctr'] = ent_kp_graph.nodes()[noun_token].get('ether_meet_ctr',0)
            ent_kp_graph.nodes()[noun_token]['ether_grp_ctr'] = ent_kp_graph.nodes()[noun_token].get('ether_grp_ctr',0)
            ent_kp_graph.nodes()[noun_token]['ether_sent_ctr'] = ent_kp_graph.nodes()[noun_token].get('ether_sent_ctr',0)
        else:
            nouns_to_update.append(noun_token)
            ent_kp_graph.add_node(noun_token,
                                  is_ether_node=True,
                                  node_type = 'kp_token',
                                  node_label="N/A",
                                  art_ctr = 0,
                                  para_ctr = 0,
                                  sent_ctr = 0)  
    
    for kp in multi_tok_kps:
        kp_nouns = set(kp.split(' '))&set(noun_graph_tokens)
        for noun in kp_nouns:
            if ent_kp_graph.has_edge(kp,noun):
                ent_kp_graph[kp][noun]['edge_type'] = "kp_to_tok"
                ent_kp_graph[kp][noun]['ether_meet_ctr'] = ent_kp_graph[kp][noun].get('ether_meet_ctr',0) + ent_kp_graph.nodes[kp]['ether_meet_ctr']
                ent_kp_graph[kp][noun]['ether_grp_ctr'] = ent_kp_graph[kp][noun].get('ether_grp_ctr',0) + ent_kp_graph.nodes[kp]['ether_grp_ctr']
                ent_kp_graph[kp][noun]['ether_sent_ctr'] = ent_kp_graph[kp][noun].get('ether_sent_ctr',0) + ent_kp_graph.nodes[kp]['ether_sent_ctr']
            else:
                ent_kp_graph.add_edge(kp,
                                      noun,
                                      edge_type='kp_to_tok',
                                      ether_grp_ctr = ent_kp_graph.nodes[kp]['ether_grp_ctr'],
                                      ether_sent_ctr = ent_kp_graph.nodes[kp]['ether_sent_ctr'],
                                      ether_meet_ctr = ent_kp_graph.nodes[kp]['ether_meet_ctr'],
                                      art_ctr = 0,
                                      para_ctr = 0,
                                      sent_ctr = 0)

    for noun_token in nouns_to_update:
        ent_kp_graph.nodes[noun_token]['ether_meet_ctr'] = sum([d['ether_meet_ctr'] for n,d in ent_kp_graph[noun_token].items()])
        ent_kp_graph.nodes[noun_token]['ether_grp_ctr'] = sum([d['ether_grp_ctr'] for n,d in ent_kp_graph[noun_token].items()])
        ent_kp_graph.nodes[noun_token]['ether_sent_ctr'] = sum([d['ether_sent_ctr'] for n,d in ent_kp_graph[noun_token].items()])
    
    return ent_kp_graph

def store_final_graph(ent_kp_graph, artifacts_dir):
    graph_file = artifacts_dir + "kp_entity_graph.pkl"
    for n,d in ent_kp_graph.copy().nodes(data=True):
        if ent_kp_graph.nodes()[n].get('meet_freq_list'):
            ent_kp_graph.nodes()[n].pop('meet_freq_list')
        if ent_kp_graph.nodes()[n].get('art_freq_list'):
            ent_kp_graph.nodes()[n].pop('art_freq_list')
    
    pickle.dump(ent_kp_graph,open(graph_file,"wb"))
    return ent_kp_graph

def update_entity_feat_dict(ent_sent_dict, ent_feat_dict):
    for ent in ent_sent_dict:
        ent_feat = np.sum([gpt_model.get_text_feats(sent) for sent in ent_sent_dict[ent]],0)
        if "<ETHER>-"+ent in ent_feat_dict:
            ent_feat_dict["<ETHER>-"+ent] += ent_feat
        else:
            ent_feat_dict["<ETHER>-"+ent] = ent_feat
    return ent_feat_dict
