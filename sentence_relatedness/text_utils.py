import re
import json
import nltk
from bert_utils import *
from byte_pair_tokenizer import tokenize
import string,itertools
from scipy import spatial

contractions = json.load(open('contractions.txt','rb'))
contractions = contractions['contractions']

def replaceContractions(text):
    #text = text.lower()
    c_filt_text = ''
    for word in text.split(' '):
        if word in contractions:
            c_filt_text = c_filt_text+' '+contractions[word]
        else:
            c_filt_text = c_filt_text+' '+word
    return c_filt_text.strip()

def stripText(text):
    text = replaceContractions(text.lower())
    text = re.sub('(\d+[A-z]+)|(([A-z]+\d+))',' ',text) #remove alphanumeric words
    text = re.sub('-',' ', text)
    text = re.sub('\s+',' ', text)
    text = re.sub("'",' ', text)
    return text.strip()


#Key-phrase candidates
def getregexChunks(text, grammar):

    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text))
    all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                                                    for tagged_sent in tagged_sents))
    return [(ele[0], ele[1], ele[2], ctr) for ele,ctr in zip(all_chunks,range(len(all_chunks)))]

def getCandidatePhrases(text, pos_search_pattern_list=[r"""base: {(<JJ.*>*<NN.*>+<IN>)?<JJ>*<NN.*>+}"""]):
                                       #r"""nounverb:{<NN.+>+<.+>{0,2}<VB*>{1}}""",
                                       #r"""verbnoun:{<VB*>{1}<.+>{0,2}<NN.+>+}"""]):
                                       #r""" nounnoun:{<NN.+>+<.+>{1,2}<NN.+>+}"""]):
                                       #r"""baseverb: {(<JJ.+>+<IN>)?<JJ>*<VB.*>+}"""]):
    punct = set(string.punctuation)
    all_chunks = []

    for pattern in pos_search_pattern_list:
        all_chunks+=getregexChunks(text, pattern)
    
    candidate_locs = [' '.join(str(ctr) for word, pos, 
                           chunk,ctr in group).lower() 
                  for key, group in itertools.groupby(all_chunks, 
                  lambda_unpack(lambda word, pos, chunk, ctr: chunk != 'O')) if key]
    
    candidate_phrases = [' '.join(word for word, pos, 
                           chunk,ctr in group).lower() 
                  for key, group in itertools.groupby(all_chunks, 
                  lambda_unpack(lambda word, pos, chunk, ctr: chunk != 'O')) if key]
    
    #candidate_phrases = [cand for cand in candidates_tokens if cand not in stop_words and not all(char in punct for char in cand)]

    return candidate_phrases,candidate_locs

def lambda_unpack(f):
    return lambda args: f(*args)

def getKeyPhraseFeatures(kp_list, kp_loc_idx,text_feats, text_tokens):
    
    key_phrase_feats = []
    for ele,loc_list in zip(kp_list,kp_loc_idx):
        if len(ele.split(' '))==1:
            idx_val = int(loc_list[0])
            key_phrase_feats.append(getTokenFeature(ele,idx_val,text_feats,text_tokens))
        else:
            curr_feature_vec = []
            for tok,tok_idx in zip(ele.split(' '),loc_list.split(' ')):
                curr_feature_vec.append(getTokenFeature(tok,int(tok_idx),text_feats,text_tokens))
            key_phrase_feats.append(sum(curr_feature_vec))
    return key_phrase_feats

def getTokenFeature(token, token_idx, text_feats, text_tokens):    
    if text_tokens[token_idx]==token:
        feat_vec = text_feats[token_idx]
    else:
        #print('Token not found in the location, searching entire text.: ', token)
        if token in text_tokens:
            idx_val = text_tokens.index(token)
            feat_vec = text_feats[idx_val]
        else:
            #print('Token not found.. returning default feature vector: ', token)
            feat_vec = np.full(len(text_feats[0]),0.01)
    return feat_vec

def getWordFeatsFromBertTokenFeats(sent_tokens,bert_tokens,bert_token_feats):
    #steps for merging the bert tokens to get the BERT features for actual words
    #1. iterate over the BERT base tokenizer
    #2. lookup for the actual word in the current BERT lookup postions
    #3. If found:
        #3a. the word is not tokenized further - use the current BERT features as word embedding
    #else:
        #3b. the word is tokenized in BERT - find the sequence of tokens and sum up the features to get the word vector
    base_ctr = 0
    bert_ctr = 0
    word_feat_list = []

    for word in sent_tokens:
        if bert_tokens[bert_ctr] == word:#word not further tokenized, use the same feature vector
            word_feat_list.append(np.array(bert_token_feats[bert_ctr].detach().numpy()))
            base_ctr+=1
            bert_ctr+=1
        else:
            aggr_feats = np.array(bert_token_feats[bert_ctr].detach().numpy())
            aggr_word = bert_tokens[bert_ctr]
            merge_next = True
            while merge_next and bert_ctr<len(bert_tokens)-1:
                if '#' in bert_tokens[bert_ctr+1]:
                    aggr_word = aggr_word+bert_tokens[bert_ctr+1]
                    bert_ctr+=1
                    aggr_feats+=np.array(bert_token_feats[bert_ctr].detach().numpy())
                else:
                    merge_next = False
                    bert_ctr+=1
            word_feat_list.append(aggr_feats)
    assert len(sent_tokens)==len(word_feat_list)
    return word_feat_list

def getKPBasedSimilarity(model, text1, text2, bert_layer = -1):
    
    token_feats_1,final_feats1,text1_bert_tokenized = getBERTFeatures(model, text1, attn_head_idx=bert_layer)
    token_feats_2,final_feats2,text2_bert_tokenized = getBERTFeatures(model, text2, attn_head_idx=bert_layer)

    text1_sent_tokens = tokenize(text1)
    text2_sent_tokens = tokenize(text2)

    merged_feats_text1 = getWordFeatsFromBertTokenFeats(text1_sent_tokens,text1_bert_tokenized,token_feats_1)
    merged_feats_text2 = getWordFeatsFromBertTokenFeats(text2_sent_tokens,text2_bert_tokenized,token_feats_2)

    #get candidate key-phrases for both sentences
    kps_sent1,kps_loc_sent1 = getCandidatePhrases(text1)
    kps_sent2,kps_loc_sent2 = getCandidatePhrases(text2)
    
    sent1_kp_feats = getKeyPhraseFeatures(kps_sent1,kps_loc_sent1,merged_feats_text1,text1_sent_tokens)
    sent2_kp_feats = getKeyPhraseFeatures(kps_sent2,kps_loc_sent2,merged_feats_text2,text2_sent_tokens)

    curr_max = 0
    for sent1_kp, feats1 in zip(kps_sent1,sent1_kp_feats):
        for sent2_kp, feats2 in zip(kps_sent2,sent2_kp_feats):
            if len(sent1_kp)<3 or len(sent2_kp)<3:
                curr_sim = 0.1
            else:
                curr_sim = 1-spatial.distance.cosine(feats1,feats2)
            if curr_sim>curr_max:
                curr_max = curr_sim    
    return curr_max

def getKPBasedSimilarityFromBERTFeats(tup1,tup2, text1, text2, bert_layer = -1):

    #tup1, tup2 - output from getBERTFeatures() as a tuples
    
    token_feats_1,final_feats1,text1_bert_tokenized = tup1
    token_feats_2,final_feats2,text2_bert_tokenized = tup2

    text1_sent_tokens = tokenize(text1)
    text2_sent_tokens = tokenize(text2)

    merged_feats_text1 = getWordFeatsFromBertTokenFeats(text1_sent_tokens,text1_bert_tokenized,token_feats_1)
    merged_feats_text2 = getWordFeatsFromBertTokenFeats(text2_sent_tokens,text2_bert_tokenized,token_feats_2)

    #get candidate key-phrases for both sentences
    kps_sent1,kps_loc_sent1 = getCandidatePhrases(text1)
    kps_sent2,kps_loc_sent2 = getCandidatePhrases(text2)

    sent1_kp_feats = getKeyPhraseFeatures(kps_sent1,kps_loc_sent1,merged_feats_text1,text1_sent_tokens)
    sent2_kp_feats = getKeyPhraseFeatures(kps_sent2,kps_loc_sent2,merged_feats_text2,text2_sent_tokens)

    curr_max = 0
    for sent1_kp, feats1 in zip(kps_sent1,sent1_kp_feats):
        for sent2_kp, feats2 in zip(kps_sent2,sent2_kp_feats):
            if len(sent1_kp)<3 or len(sent2_kp)<3:
                curr_sim = 0.1
            else:
                curr_sim = 1-spatial.distance.cosine(feats1,feats2)
            if curr_sim>curr_max:
                curr_max = curr_sim

    return curr_max

def getCosineSimilarity(text1, text2,  bert_layer = -1):

    token_feats_1,final_feats1,text1_bert_tokenized = getBERTFeatures(model, text1, attn_head_idx=layer)
    token_feats_2,final_feats2,text2_bert_tokenized = getBERTFeatures(model, text2, attn_head_idx=layer)

    return 1-spatial.distance.cosine(final_feats1,final_feats2)