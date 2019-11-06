# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Arjun
#     language: python
#     name: arjun
# ---

from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers import WEIGHTS_NAME, BertConfig, BertForTokenClassification, BertTokenizer
import torch
import re
from nltk import sent_tokenize

class BERT_NER():
    '''
    USAGE:
            ner_model = BERT_NER('../models/bert-ner-uncased-2/',cased=False)
            text = "the city new york has a beautiful skyline around central park. I would like to see it some day."
            entities, confidence_scores = ner_model.get_entities(text)
    '''
    def __init__(self, model_path,
                 labels = ["O","E"], cased=False):
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case = not cased)
        self.sm = torch.nn.Softmax(dim=1)
        self.labels = labels
        self.contractions = {"ain't": 'am not', "aren't": 'are not', "can't": 'cannot', "can't've": 'cannot have', "'cause": 'because', "could've": 'could have', "couldn't": 'could not', "couldn't've": 'could not have', "didn't": 'did not', "doesn't": 'does not', "don't": 'do not', "hadn't": 'had not', "hadn't've": 'had not have', "hasn't": 'has not', "haven't": 'have not', "he'd": 'he would', "he'd've": 'he would have', "he'll": 'he will', "he's": 'he is', "how'd": 'how did', "how'll": 'how will', "how's": 'how is', "i'd": 'i would', "i'll": 'i will', "i'm": 'i am', "i've": 'i have', "isn't": 'is not', "it'd": 'it would', "it'll": 'it will', "it's": 'it is', "let's": 'let us', "ma'am": 'madam', "mayn't": 'may not', "might've": 'might have', "mightn't": 'might not', "must've": 'must have', "mustn't": 'must not', "needn't": 'need not', "oughtn't": 'ought not', "shan't": 'shall not', "sha'n't": 'shall not', "she'd": 'she would', "she'll": 'she will', "she's": 'she is', "should've": 'should have', "shouldn't": 'should not', "that'd": 'that would', "that's": 'that is', "there'd": 'there had', "there's": 'there is', "they'd": 'they would', "they'll": 'they will', "they're": 'they are', "they've": 'they have', "wasn't": 'was not', "we'd": 'we would', "we'll": 'we will', "we're": 'we are', "we've": 'we have', "weren't": 'were not', "what'll": 'what will', "what're": 'what are', "what's": 'what is', "what've": 'what have', "where'd": 'where did', "where's": 'where is', "who'll": 'who will', "who's": 'who is', "won't": 'will not', "wouldn't": 'would not', "you'd": 'you would', "you'll": 'you will', "you're": 'you are'}
    
    def replaceContractions(self,text):
        c_filt_text = ''
        for word in text.split(' '):
            if word.lower() in self.contractions:
                c_filt_text = c_filt_text+self.contractions[word.lower()]+' '
            else:
                c_filt_text = c_filt_text+word+' '
        return c_filt_text.strip()
    
    def get_entities(self,text):
        segment_entities=[]
        segment_scores=[]
        for sent in sent_tokenize(text):
            sent_ent, sent_score= self.get_entities_from_sentence(sent)
            
            segment_entities.extend(sent_ent)
            segment_scores.extend(sent_score)
        
        # removing duplicate entities
        seg_entities = dict(zip(segment_entities,segment_scores))
        seg_words = list(seg_entities.keys())
        seg_scores = list(seg_entities.values())
        return seg_words, seg_scores
            
    def get_entities_from_sentence(self,text):
        input_ids=[]
        tokenized_text = self.tokenize(text)
        token_to_word=[]
        
        # cleaning and splitting text, preserving punctuation
        clean_text = self.replaceContractions(text)
        split_text = re.split("[\s]|([?.,!/]+)",clean_text)
        
        for word in split_text:
            if word not in ['',None]:
                toks = self.tokenizer.encode(word)
                # removing characters that usually do not appear within text
                clean_word =re.sub(r'[^a-zA-Z0-9_\'*-]+','',word).strip()
                token_to_word.extend([clean_word]*len(toks))
                input_ids.extend(toks)
                
        # Calculating batch size based on nearest "." from mid-point of text if length exceeds 512
        if len(input_ids)>512:
            mid = len(input_ids)//2
            batch_size = mid - 1 - input_ids[:mid][::-1].index(self.tokenizer.encode(".")[0])
        else:
            batch_size = 510 
            
        entities=[]
        potential_entities=[]
        for i in range(0,len(input_ids),batch_size):
            encoded_text_sp = self.tokenizer.encode("[CLS]") + input_ids[i:i+batch_size] + self.tokenizer.encode("[SEP]")
            input_ids = torch.tensor(encoded_text_sp).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(input_ids)[0][0,1:-1]
            for j,(tok,embed) in enumerate(zip(token_to_word[i:i+batch_size],list(outputs))):
                embed=embed.unsqueeze(0)
                score = self.sm(embed).detach().numpy().max(-1)[0]
                label = self.labels[self.sm(embed).argmax().detach().numpy()]
                # Consider Entities and Non-Entities with low confidence (false negatives)
                if label!="O" or (label=="O" and score<0.99):
                    entities.append((tok,score))
        
        sent_entity_list, sent_scores = self.concat_entities(clean_text,entities)
        return sent_entity_list, sent_scores
    
    def tokenize(self,text):
        return self.tokenizer.tokenize(text)
    
    def concat_entities(self,text,entities):
        sent_entity_list=[]
        sent_scores=[]
        seen=[]
        # handling abbreviations such as U.S.
        text = re.sub("[A-Z][.]\s?[A-Z][.]?",lambda mobj: mobj.group(0)[0] + " " + mobj.group(0)[-2],text).casefold()
        # remove consecutive duplicate entities
        entity_words = list(dict.fromkeys([e[0] for e in entities if e[0]!='']))
        entity_scores = {e[0]:e[1] for e in entities}
        
        for i in range(len(entity_words)):
            if i in seen:
                continue
            if text.count(entity_words[i].casefold())>=1:
                conc = entity_words[i].strip("'\"")+" "
                conc = conc if conc[0].isupper() else conc.capitalize()
                check = entity_words[i]+" "
                score = entity_scores[entity_words[i]]
                k=i+1
                seen+=[i]
                while k<len(entity_words) and text.count(check.casefold()+entity_words[k].casefold())>=1:
                    conc_word=entity_words[k].strip("'\"")+" "
                    conc += conc_word if conc_word[0].isupper() else conc_word.capitalize()
                    check+= entity_words[k]+" "
                    seen+=[k]
                    score += entity_scores[entity_words[k]]
                    k+=1
                sent_entity_list += [conc.strip(" ,.")]
                sent_scores += [score/(k-i)]
        
        return sent_entity_list,sent_scores












