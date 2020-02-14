import networkx as nx
import numpy as np
from nltk.tokenize import sent_tokenize
import json
import nltk, string,itertools
from nltk.corpus import stopwords
import itertools
import pickle
import re
import random
from extra_preprocess import preprocess_text

WEB_URL_REGEX = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
stop_words = set(stopwords.words('english'))
stop_words_spacy = list("""
a about above across after afterwards again against all almost alone along
already also although always am among amongst amount an and another any anyhow
anyone anything anyway anywhere are around as at

back be became because become becomes becoming been before beforehand behind
being below beside besides between beyond both bottom but by

call can cannot ca could

did do does doing done down due during

each eight either eleven else elsewhere empty enough even ever every
everyone everything everywhere except

few fifteen fifty first five for former formerly forty four from front full
further

get give go

had has have he hence her here hereafter hereby herein hereupon hers herself
him himself his how however hundred

i if in indeed into is it its itself

keep

last latter latterly least less

just

made make many may me meanwhile might mine more moreover most mostly move much
must my myself

name namely neither never nevertheless next nine no nobody none noone nor not
nothing now nowhere n't

of off often on once one only onto or other others otherwise our ours ourselves
out over own

part per perhaps please put

quite

rather re really regarding

same say see seem seemed seeming seems serious several she should show side
since six sixty so some somehow someone something sometime sometimes somewhere
still such

take ten than that the their them themselves then thence there thereafter
thereby therefore therein thereupon these they third this those though three
through throughout thru thus to together too top toward towards twelve twenty
two

under until up unless upon us used using

various very via was we well were what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would

yet you your yours yourself yourselves yeah okay
""".split()
)
stop_words = set(list(stop_words)+list(stop_words_spacy))

class CandidateKPExtractor(object):

    def __init__(self, stop_words, filter_small_sents = True):

        self.punct = set(string.punctuation)
        self.filter_small_sents = filter_small_sents
        self.stop_words = stop_words

        
    def __init__(self, filter_small_sents = True):

        self.punct = set(string.punctuation)
        self.filter_small_sents = filter_small_sents
        self.stop_words = stop_words

    def get_candidate_phrases(self, text, pos_search_pattern_list=[r"""base: {(<JJ.*>*<NN.*>+<IN>)?<JJ>*<NN.*>+}"""]):
        all_chunks = []

        for pattern in pos_search_pattern_list:
            all_chunks+=self.getregexChunks(text, pattern)

        candidates_tokens = [' '.join(word for word, pos, 
                    chunk in group)
                    for key, group in itertools.groupby(all_chunks, 
                    self.lambda_unpack(lambda word, pos, chunk: chunk != 'O')) if key]

        candidate_phrases = [cand for cand in candidates_tokens if cand not in self.stop_words and not all(char in self.punct for char in cand)]

        return candidate_phrases

    def getregexChunks(self,text, grammar):

        chunker = nltk.chunk.regexp.RegexpParser(grammar)
        tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
        all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                        for tagged_sent in tagged_sents))
        return all_chunks

    def lambda_unpack(self, f):
        return lambda args: f(*args)
    
def get_kp_nouns(kp,sent_nouns):
    return set(kp.lower().split(' '))&set(sent_nouns)

def get_candidate_phrases(text, 
                          pos_search_pattern_list=[r"""base: {(<JJ.*>*<NN.*>+<IN>)?<JJ>*<NN.*>+}"""]):
        all_chunks = []

        for pattern in pos_search_pattern_list:
            all_chunks+=getregexChunks(text, pattern)

        candidates_tokens = [' '.join(word for word, pos, 
                    chunk in group)
                    for key, group in itertools.groupby(all_chunks, 
                                        lambda_unpack(lambda word, pos, chunk: chunk != 'O')) if key]

        candidate_phrases = [cand for cand in candidates_tokens if cand not in stop_words and not all(char in punct for char in cand)]

        return candidate_phrases

def getregexChunks(text, grammar):

    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    print(all_chunks)
    return all_chunks

def lambda_unpack(f):
    return lambda args: f(*args)

def get_freq_score(text_list, se_graph):
    filtered_kp_ctr = []
    text_kps_list = []
    kp_e = CandidateKPExtractor(stop_words)
    for input_text in text_list:
        text_kps = kp_e.get_candidate_phrases(input_text)
        text_kps = list(set([ele.lower() for ele in text_kps]))

        segment_noun_list = []
        tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(input_text))
        text_nouns = []
        for tagged_sent in tagged_sents:
            text_nouns.extend([ele[0] for ele in list(tagged_sent) if ele[1].startswith('NN')])
        text_nouns = [ele.lower() for ele in text_nouns]

        intersecting_nouns = list(set(text_nouns)&set(se_graph))

        #connected phrases
        intersection_ctr = 0
        filtered_kps = []
        for kp in text_kps:
            if len(kp.split(' '))>1:
                kp_nouns = list(set(kp.split(' '))&set(intersecting_nouns))
                for noun in kp_nouns:
                    rem_nouns = list(set(kp_nouns)-set([noun]))
                    if set(rem_nouns)&set(se_graph[noun])==set(rem_nouns):
                        filtered_kps.append(kp)
                        continue
        filtered_kps = list(set(filtered_kps))
        filtered_kp_ctr.append(len(filtered_kps))
        text_kps_list.append(filtered_kps)
    return filtered_kp_ctr



