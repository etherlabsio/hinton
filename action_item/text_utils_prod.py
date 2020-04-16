import string
import itertools
import nltk
#from bert_pos_tagger import BertPosTagger as bpt
#btag = bpt('/home/shubham/Project/pos_tag/models/epoch-2/pytorch_model.bin')
from distilBiLstm_pos_tagger import DistilBiLstmPosTagger
dblstmtag =DistilBiLstmPosTagger()
#def replaceContractions()

contractions = {"ain't": "am not",
"aren't": "are not",
"can't": "can not",
"can't've": "can not have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are",
"I'll": "I will",
"We'll": "We will",
"Let's": "Let us",
"I'd": "I would"}

negating_words = ["cannot", "not"]
present_signifier_words = ["just","now"]
drop_md_vb_words = ['see','be','would']
drop_subj_verbs = ['tell','talk','walk']

def replace_contractions(text):
    c_filt_text = ''
    for word in text.split(' '):
        if word in list(contractions):
            c_filt_text = c_filt_text+' '+contractions[word]
        else:
            c_filt_text = c_filt_text+' '+word
    return c_filt_text
    
class CandidateKPExtractor(object):
    def __init__(self, stop_words, filter_small_sents=True):

        self.punct = set(string.punctuation)
        self.filter_small_sents = filter_small_sents
        self.stop_words = stop_words
        
    def get_candidate_phrases(self,text,
        pos_search_pattern_list=[
            r"""verbnoun:{<MD|VB>+<.+>{0,2}<VB>+<.+>{0,2}<NN.*>+(<.+>{0,2}<JJ.*>*<NN.*>+)*}"""],call_ctr = 0):
        all_chunks = []

        for pattern in pos_search_pattern_list:
            all_chunks += self.getregexChunks(text, pattern)

        candidates_tokens = [
            " ".join(word for word, pos, chunk in group).lower()
            for key, group in itertools.groupby(
                all_chunks, self.lambda_unpack(lambda word, pos, chunk: chunk != "O"),
            )
            if key
        ]
        candidate_phrases = [
            cand
            for cand in candidates_tokens
            if cand not in self.stop_words
            and not all(char in self.punct for char in cand)
        ]
        candidate_phrases = [self.fixRegexPattern(candidate) for candidate in candidate_phrases]
        if len(candidate_phrases)==0 and call_ctr==0:
            #search by extending the context
            extended_context_flag=True
            candidate_phrases = self.get_candidate_phrases(
                text,
                pos_search_pattern_list=[
            r"""verbnoun:{<MD|VB>+<.+>{0,2}<VB>+<.+>{0,5}<NN.*>+(<.+>{0,2}<JJ.*>*<NN.*>+)*}"""],
                            call_ctr = 1)
            
        return candidate_phrases

    def get_subject_candidates(self, text):
        candidate_phrases = self.get_candidate_phrases(text)
        candidate_phrases = self.filter_phrases_by_word_lookup(text, candidate_phrases, present_signifier_words)
        candidate_phrases = [candidate for candidate in candidate_phrases if candidate.split(' ')[0] not in drop_md_vb_words]
        candidate_subject_list = []
        #text to search - input text starting from the start of first candidate phrase
        if len(candidate_phrases)>0:
            candidate_phrase_idx = text.find(candidate_phrases[0])
            text = text[candidate_phrase_idx:]
            pos_search_pattern_list=[r"""verbnoun:{<VB>+<.+>{0,2}<NN.*>+(<.+>{0,2}<JJ.*>*<NN.*>+)*}"""]
            candidate_subject_list+=self.get_candidate_phrases(text,pos_search_pattern_list,call_ctr = 1)
            if len(candidate_subject_list)==0:
                pos_search_pattern_list=[r"""verbnoun:{<VB>+<.+>{0,5}<NN.*>+(<.+>{0,2}<JJ.*>*<NN.*>+)*}"""]
                candidate_subject_list+=self.get_candidate_phrases(text,pos_search_pattern_list,call_ctr = 1)
        candidate_subject_list = [candidate for candidate in candidate_subject_list if candidate.split(' ')[0] not in drop_subj_verbs]
        candidate_subject_list = [candidate for candidate in candidate_subject_list if candidate.split(' ')[0] not in drop_md_vb_words]
        return candidate_subject_list

    def get_ai_subjects(self,text,filter_neg_subjs=True):
        text = replace_contractions(text.lower())
        
        if 'i' in text.split(' '): #hot fix to handle to handle NLTK regression
            text_split = text.split(' ')
            text_split = ['I' if x=='i' else x for x in text_split]
            text = ' '.join(text_split).strip()
            
        ai_candidates = self.get_subject_candidates(text)
        #print('ai_candidates',ai_candidates)
        if filter_neg_subjs:
            ai_candidates = self.filter_phrases_by_word_lookup(text,ai_candidates,negating_words,1)
        return ai_candidates
        
    def getregexChunks(self, text, grammar,pos_name = 'bilstm'):

        chunker = nltk.chunk.regexp.RegexpParser(grammar)
        if(pos_name == 'bilstm'):
            print('using bilstm')
            tagged_sents = [dblstmtag.get_sent_pos_tags(t) for t in nltk.sent_tokenize(text) ]
        else:
            tagged_sents = nltk.pos_tag_sents(
                nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(text)
            )
        #tagged_sents = [btag.get_sent_pos_tags(t) for t in nltk.sent_tokenize(text) ]
        #tagged_sents = [dblstmtag.get_sent_pos_tags(t) for t in nltk.sent_tokenize(text) ]
        #print(tagged_sents)
        all_chunks = list(
            itertools.chain.from_iterable(
                nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
                for tagged_sent in tagged_sents
            )
        )
        return all_chunks

    def lambda_unpack(self, f):
        return lambda args: f(*args)

    def fixRegexPattern(self,text):    
        filtered_tok_list = []
        for i in range(len(text.split(' '))):
            tok = text.split(' ')[i]
            if tok[0] == "'":
                if len(filtered_tok_list)>0:
                    merge_tok = filtered_tok_list.pop()
                    filtered_tok_list.append(merge_tok+tok)
            else:
                filtered_tok_list.append(tok)
        return(' '.join(filtered_tok_list))

    def filter_phrases_by_word_lookup(self,input_sentence, subject_list, drop_words, neg_check = 0):
    #for each find the
        input_sentence = input_sentence.replace('.','')
        input_sentence = input_sentence.replace(',','')
        input_sentence = input_sentence.replace('?','').lower()
        filtered_subjects_list = []
        input_string_split = input_sentence.split(' ')
        negation_search_window = 3
        
        for candidate_subject in subject_list:
            candidate_split = candidate_subject.split(' ')
            candidate_start_locs = [enum for enum,ele in enumerate(input_string_split) if ele == candidate_split[0]]
            candidate_end_locs = [enum for enum,ele in enumerate(input_string_split) if ele == candidate_split[-1]]
            for start_loc in candidate_start_locs:
                for end_loc in candidate_end_locs:
                    if (input_string_split[start_loc:end_loc+1])==candidate_split:
                        #check any of the previous elements contain negating words
                        search_start_idx = max(0,start_loc-negation_search_window)
                        
                        if len(set(input_string_split[search_start_idx:start_loc]) & set(drop_words))==0:
                            filtered_subjects_list.append(candidate_subject)
                        elif(neg_check==1):
                            pos_search_pattern_list=[r"""negation:{(<.*><RB>)+<.+>{0,2}<VB>+<.+>{0,2}<NN.*>+(<.+>{0,2}<JJ.*>*<NN.*>+)*}"""]
                            filtered_subjects_list+=self.get_candidate_phrases(input_sentence,pos_search_pattern_list,call_ctr = 1)

        return list(filtered_subjects_list)
