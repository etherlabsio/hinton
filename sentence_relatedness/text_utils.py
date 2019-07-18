import re
import json
import nltk
from bert_utils import *
from byte_pair_tokenizer import tokenize
import string,itertools
from scipy import spatial
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from itertools import chain

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

def getCandidatePhrases(text, pos_search_pattern_list=[r"""base: {(<JJ.*>*<NN.*>+<IN>)?<JJ>*<NN.*>+}""",
                                           r"""nounverb:{<NN.*>+<VB.*>+}""",
                                           r"""verbnoun:{<VB.*>+<NN.*>+}"""]):
                                       #r""" nounnoun:{<NN.+>+<.+>{1,2}<NN.+>+}"""]):
                                       #r"""baseverb: {(<JJ.+>+<IN>)?<JJ>*<VB.*>+}"""]):
    text = stripText(text)
    punct = set(string.punctuation)
    all_chunks = []

    candidate_phrases = []
    for pattern in pos_search_pattern_list:
        curr_chunks=getregexChunks(text, pattern)
        candidate_phrases+=[' '.join(word for word, pos, 
                           chunk,ctr in group).lower() 
                  for key, group in itertools.groupby(curr_chunks, 
                  lambda_unpack(lambda word, pos, chunk, ctr: chunk != 'O')) if key]
    
    filtered_candidates = []
    for key_phrase in candidate_phrases:
        curr_filtr_phrase = stripStopWordsFromText(key_phrase,stop_words)
        if len(curr_filtr_phrase)>0:
            filtered_candidates.append(curr_filtr_phrase)
    candidate_phrases = filterCandidatePhrases(text,filtered_candidates)
    candidate_phrases,candidate_locs = getPhraseListLocations(text, candidate_phrases)

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

def getKPBasedSimilarity(text1,text2,model,layer = -1):

    text1 = stripText(text1)
    text2 = stripText(text2)

    token_feats_1,final_feats1,text1_bert_tokenized = getBERTFeatures(model, text1, attn_head_idx=layer)
    token_feats_2,final_feats2,text2_bert_tokenized = getBERTFeatures(model, text2, attn_head_idx=layer)

    text1_sent_tokens = tokenize(text1)
    text2_sent_tokens = tokenize(text2)

    merged_feats_text1 = getWordFeatsFromBertTokenFeats(text1_sent_tokens,text1_bert_tokenized,token_feats_1)
    merged_feats_text2 = getWordFeatsFromBertTokenFeats(text2_sent_tokens,text2_bert_tokenized,token_feats_2)

    #get candidate key-phrases for both sentences
    kps_sent1,kps_loc_sent1 = getCandidatePhrases(text1)
    kps_sent2,kps_loc_sent2 = getCandidatePhrases(text2)
    
    sent1_kp_feats = getKeyPhraseFeatures(kps_sent1,kps_loc_sent1,merged_feats_text1,text1_sent_tokens)
    sent2_kp_feats = getKeyPhraseFeatures(kps_sent2,kps_loc_sent2,merged_feats_text2,text2_sent_tokens)

    sim_list = []
    for sent1_kp, feats1 in zip(kps_sent1,sent1_kp_feats):
        for sent2_kp, feats2 in zip(kps_sent2,sent2_kp_feats):
            if len(sent1_kp.split(' '))+len(sent2_kp.split(' '))==2:
                if len(sent1_kp.split(' ')[0])<4 or len(sent2_kp.split(' ')[0])<4:
                    continue
                curr_sim = 1-spatial.distance.cosine(feats1,feats2)
                print(sent1_kp,'<>',sent2_kp,': ',curr_sim)
            else:
                if len(sent1_kp.split(' '))==1 or len(sent2_kp.split(' '))==1:
                    print('Skipping: ',sent1_kp,sent2_kp )
                    continue
                else:
                    curr_sim = 1-spatial.distance.cosine(feats1,feats2)
                    print(sent1_kp,'<>',sent2_kp,': ',curr_sim)
            sim_list.append(curr_sim)

    if len(sim_list)>3:
        sim_list = sim_list[0:3]
        
    mean_dist = np.mean(sim_list)
        
    return mean_dist

def getKPBasedSimilarityFromBERTFeats(tup1,tup2, text1, text2, bert_layer = -1):

    #tup1, tup2 - output from getBERTFeatures() as a tuples

    text1 = stripText(text1)
    text2 = stripText(text2)
    
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

    sim_list = []
    for sent1_kp, feats1 in zip(kps_sent1,sent1_kp_feats):
        for sent2_kp, feats2 in zip(kps_sent2,sent2_kp_feats):
            if len(sent1_kp.split(' '))+len(sent2_kp.split(' '))==2:
                if len(sent1_kp.split(' ')[0])<4 or len(sent2_kp.split(' ')[0])<4:
                    continue
                curr_sim = 1-spatial.distance.cosine(feats1,feats2)
                print(sent1_kp,'<>',sent2_kp,': ',curr_sim)
            else:
                if len(sent1_kp.split(' '))==1 or len(sent2_kp.split(' '))==1:
                    print('Skipping: ',sent1_kp,sent2_kp )
                    continue
                else:
                    curr_sim = 1-spatial.distance.cosine(feats1,feats2)
                    print(sent1_kp,'<>',sent2_kp,': ',curr_sim)
            sim_list.append(curr_sim)

    if len(sim_list)>3:
        sim_list = sim_list[0:3]
        
    mean_dist = np.mean(sim_list)
        
    return mean_dist

def getCosineSimilarity(text1,text2,model,layer = -1):

    token_feats_1,final_feats1,text1_bert_tokenized = getBERTFeatures(model, text1, attn_head_idx=layer)
    token_feats_2,final_feats2,text2_bert_tokenized = getBERTFeatures(model, text2, attn_head_idx=layer)

    return 1-spatial.distance.cosine(final_feats1,final_feats2)

## Added on: 17-Jul-2019

def removeStopwords(text):
    sent = ' '.join([tok for tok in text.split(' ') if tok not in stop_words])
    return sent

def getStartEndPOSList(text,candidate_phrases_list):
    start_pos_list = []
    end_pos_list = []
    processed_list = []
    for candidate in candidate_phrases_list:
        start_pos = [match.start() for match in re.finditer(candidate, text)]
        if len(start_pos)==1:
            processed_list.append(candidate)
            start_pos_list.append(start_pos[0])
            end_pos_list.append(start_pos[0]+len(candidate))
        else:
            tok_ctr = processed_list.count(candidate)
            start_pos_list.append(start_pos[tok_ctr])
            end_pos_list.append(start_pos[tok_ctr]+len(candidate))
            processed_list.append(candidate)
    return start_pos_list, end_pos_list

def filterCandidatePhrases(text, candidate_phrases_list):
    drop_list = []
    merge_list = []
    merge_list_start = []
    merge_list_end = []

    filtered_sent = removeStopwords(text)
    filtered_phrase_list = [removeStopwords(phrase) for phrase in candidate_phrases_list]

    start_pos_list, end_pos_list = getStartEndPOSList(text,candidate_phrases_list)
    filtered_start_pos_list, filtered_end_pos_list = getStartEndPOSList(filtered_sent,filtered_phrase_list)
    assert len(filtered_start_pos_list)==len(filtered_phrase_list)

    for i in range(len(start_pos_list)):
        curr_start,curr_end,ctr = start_pos_list[i],end_pos_list[i],i

        for j in range(i+1, len(start_pos_list)):
            lookup_start, lookup_end, lookup_ctr = start_pos_list[j], end_pos_list[j], j
            if curr_start==lookup_start and curr_end==lookup_end:
                continue
            if (curr_start<=lookup_start and curr_end>=lookup_end) or (lookup_start<=curr_start and lookup_end>=curr_end):
                if len(candidate_phrases_list[i])<len(candidate_phrases_list[j]):
                    drop_list.append(candidate_phrases_list[i])
                else:
                    drop_list.append(candidate_phrases_list[j])

        for k in range(len(start_pos_list)):
            if filtered_start_pos_list[i]-filtered_end_pos_list[k]==1:
                merge_list.append([candidate_phrases_list[i],candidate_phrases_list[k]])
                drop_list.append(candidate_phrases_list[i])
                drop_list.append(candidate_phrases_list[k])
                merge_list_start.append(min(start_pos_list[i],start_pos_list[k]))
                merge_list_end.append(max(end_pos_list[i],end_pos_list[k]))

    for ctr in range(len(merge_list)):
        candidate_phrases_list.append(text[merge_list_start[ctr]:merge_list_end[ctr]])
        
    doup_list = []   
    for kp1 in candidate_phrases_list:
        for kp2 in candidate_phrases_list:
            if kp1!=kp2 and kp2 in kp1:
                doup_list.append(kp2)
    #do not do set operation            
    for ele in drop_list:
        if ele in candidate_phrases_list:
            candidate_phrases_list.remove(ele)
            
    for ele in doup_list:
        if ele in candidate_phrases_list:
            candidate_phrases_list.remove(ele)

    return candidate_phrases_list

def stripStopWordsFromText(sent, stop_words):
    fw_ctr = 0
    bw_ctr = 0
    for tok in sent.split(' '):
        if tok in stop_words:
            fw_ctr+=1
        else:
            break
    for tok in reversed(sent.split(' ')):
        if tok in stop_words:
            bw_ctr-=1
        else:
            break
    if bw_ctr!=0:
        stripped_kp = ' '.join(sent.split(' ')[fw_ctr:bw_ctr])
    else:
        stripped_kp = ' '.join(sent.split(' ')[fw_ctr:])
            
    return stripped_kp.strip()

def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))        
    range_list = [list(range(ele[0],ele[1]+1)) for ele in results]
    
    return range_list

def getPhraseListLocations(text, candidate_phrases):
    #assuming that the 
    phrase_idx_list = []
    token_sent_list = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text)]
    token_list = list(chain(*token_sent_list))
    
    for phrase in candidate_phrases:
        phrase_tokens = nltk.word_tokenize(phrase)
        phrase_idx = find_sub_list(phrase_tokens,token_list)
        phrase_idx_list.append(phrase_idx)
     
    processed_phrase_list = []
    processed_idx_list = []
    for phrase, loc_idx in zip(candidate_phrases,phrase_idx_list):
        if len(loc_idx)==1:
            processed_phrase_list.append(phrase)
            processed_idx_list.append(loc_idx[0])
        else:
            #count number of times the phrase has occurred in the list
            if phrase not in processed_phrase_list:
                kp_occ_ctr = candidate_phrases.count(phrase)
                if kp_occ_ctr == len(loc_idx):
                    #append current key-phrase `kp_occ_ctr` times into the lists
                    processed_phrase_list+=[phrase]*kp_occ_ctr
                    processed_idx_list+=loc_idx
                else: 
                    idx_drop_list = []
                    #the phrase index is calculated as part of another key-phrase index
                    #check other sublists that are 
                    #find other locations 
                    for lookup_loc in phrase_idx_list:
                        if lookup_loc!=loc_idx and len(lookup_loc[0])!=len(loc_idx[0]):
                            for i in range(len(curr)):
                                if((set(loc_idx[i]) & set(lookup_loc[0]))== set(loc_idx[i])):
                                    idx_drop_list.append(loc_idx[i])
                    for to_insert_loc in loc_idx:
                        if to_insert_loc not in idx_drop_list:
                            processed_phrase_list.append(phrase)
                            processed_idx_list.append(to_insert_loc)
                            
    str_loc_list = []
    for ele in processed_idx_list:
        str_loc = ''
        for tok in ele:
            str_loc = str_loc+' '+str(tok)
        str_loc_list.append(str_loc.strip())
            
    return processed_phrase_list,str_loc_list
