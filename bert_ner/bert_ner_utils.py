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

class BERT_NER():
    '''
    USAGE:
            ner_model = BERT_NER('../models/bert-ner/',cased=False)
            ner_model_c = BERT_NER('../models/bert-ner-cased/',cased=True)
            text = "the city new york has a beautiful skyline around central park."
            entity_tokens, confidence_scores = ner_model.get_entities(text)
            entities = ner_model.wordize(entity_tokens, capitalize=True)
            
            For Non-Entity Analysis:
            ent_tokens, ent_conf, non_ent_tok, non_ent_conf = ner_model_c.get_entities(text, get_non_entities=True)
            
    '''
    def __init__(self, model_path,device="cpu",
                 labels = ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"],cased=False):
        self.device = torch.device(device)
        self.model = BertForTokenClassification.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(model_path,do_lower_case = not cased)
        self.sm = torch.nn.Softmax(dim=1)
        self.labels = labels

    def get_entities(self, text, get_non_entities=False):
        # input: text - str
        # output: entity tokens - list(str), confidence scores - list(float)
        scores=[]
        labels=[]
        tokenized_text = self.tokenize(text)
        encoded_text = self.tokenizer.encode(text)
        
        # Calculating batch size based on nearest "." from mid-point of text if length exceeds 512
        if len(encoded_text)>512:
            mid = len(encoded_text)//2
            batch_size = mid - 1 - encoded_text[:mid][::-1].index(self.tokenizer.encode(".")[0])
        else:
            batch_size = 510 
        
        for i in range(0,len(encoded_text),batch_size):
            encoded_text_sp = self.tokenizer.encode("[CLS]") + encoded_text[i:i+batch_size] + self.tokenizer.encode("[SEP]")
            input_ids = torch.tensor(encoded_text_sp).unsqueeze(0)
            outputs = self.model(input_ids)[0][0,1:-1]
            
            scores.extend(list(self.sm(outputs).max(-1)[0].detach().numpy()))
            labels.extend(list([self.labels[y] for y in self.sm(outputs).argmax(-1).detach().numpy()]))
        
        entities=[]
        non_entities = []
        non_conf=[]
        conf = []
        seen=[]
        
        for ind,(tok_text,label) in enumerate(zip(tokenized_text,labels)):
            if label=="O" and ind not in seen:
                # Check for false negative (low confidence of "O" label)
                confidence_thresh = 0.95
                size_thresh=1
                if scores[ind] < confidence_thresh and len(tok_text)>size_thresh:
                    label="E"
                    scores[ind] = 1-scores[ind]
                else:
                    non_entities.append(tok_text)
                    non_conf.append(scores[ind])
                
            if label!="O" and ind not in seen:
                # If current token is mid-token(starts with #), go back till you find source token
                if '##' in tok_text: 
                    k=1
                    tmp=[]
                    insert_pos = len(entities) if len(entities)>=1 else 0
                    while(tokenized_text[ind-k][0]=="#"):
                        conf.insert(insert_pos,scores[ind-k])
                        entities.insert(insert_pos,tokenized_text[ind-k])
                        seen.append(ind-k)
                        k+=1
                    
                    seen.append(ind-k)
                    conf.insert(insert_pos,scores[ind-k]) 
                    entities.insert(insert_pos,tokenized_text[ind-k])

                # Append current token
                entities.append(tok_text)
                conf.append(scores[ind])
                seen.append(ind)

                # Check for additional tokens and append
                if ind < len(tokenized_text)-1:
                    if tokenized_text[ind+1][0]=="#":
                        k=1
                        while ind+k <len(tokenized_text) and (tokenized_text[ind+k][0]=="#"):
                            entities.append(tokenized_text[ind+k])
                            seen.append(ind+k)
                            conf.append(scores[ind+k])
                            k+=1

        if get_non_entities:
            return entities, conf, non_entities, non_conf
        else:
            return entities, conf

    def tokenize(self,text):
        return self.tokenizer.tokenize(text)

    def wordize(self,tokens, capitalize=False):
        words =  self.tokenizer.convert_tokens_to_string(tokens)
        if capitalize:
            return list(map(lambda x: x.capitalize(),words.split()))
        else:
            return list(map(lambda x: x.upper(),words.split()))
