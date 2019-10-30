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
    def get_entities(self,text,get_non_entities=False):
        input_ids=[]
        tokenized_text = self.tokenize(text)
        token_to_word=[]
        
        for word in text.split():
            toks = self.tokenizer.encode(word)
            token_to_word.extend([re.sub(r'[^a-zA-Z0-9_\'"*-]+','', word.lower())]*len(toks))
            input_ids.extend(toks)
        # Calculating batch size based on nearest "." from mid-point of text if length exceeds 512
        if len(input_ids)>512:
            mid = len(input_ids)//2
            batch_size = mid - 1 - input_ids[:mid][::-1].index(self.tokenizer.encode(".")[0])
        else:
            batch_size = 510 
        entities={}
        non_entities={}
        for i in range(0,len(input_ids),batch_size):
            encoded_text_sp = self.tokenizer.encode("[CLS]") + input_ids[i:i+batch_size] + self.tokenizer.encode("[SEP]")
            input_ids = torch.tensor(encoded_text_sp).unsqueeze(0)
            
            with torch.no_grad():
                outputs = self.model(input_ids)[0][0,1:-1]
            for j,(tok,embed) in enumerate(zip(token_to_word[i:i+batch_size],list(outputs))):
                embed=embed.unsqueeze(0)
                score = self.sm(embed).detach().numpy().max(-1)[0]
                label = self.labels[self.sm(embed).argmax().detach().numpy()] 
                if label!="O" or (label=="O" and score<0.98):
                    entities[tok] = max(entities.get(tok,0),score)
                else:
                    non_entities[tokenized_text[j]] = score
        final_entity_list, final_scores = self.concat_entities(text,entities)
        if get_non_entities:
            return final_entity_list, final_scores,list(non_entities.keys()),list(non_entities.values())
        return final_entity_list, final_scores
    def tokenize(self,text):
        return self.tokenizer.tokenize(text)

    def wordize(self,entities, capitalize=False):
        if capitalize:
            return entities
        else:
            return "|".join(entities).casefold().split("|")
    
    def concat_entities(self,text,entities):
        final_entity_list=[]
        final_scores=[]
        seen=[]
        text = re.sub("[A-Z][.] ?[A-Z]",lambda x: x.group(0)[0]+" "+x.group(0)[-1],text)
        split_text = [re.sub(r'[^a-zA-Z0-9_\'"*-,?!.]+','',w.lower()) for w in re.split("[ ]|([?.,!]+)",text) if w is not None]
        for i in range(len(split_text)):
            if i in seen:
                continue
            if split_text[i] in entities:
                conc = split_text[i].strip("'\"").capitalize()+" "
                score = entities[split_text[i]]
                k=i+1
                seen+=[i]
                while k<len(split_text) and split_text[k] in entities:
                    conc+=split_text[k].strip("'\"").capitalize()+" "
                    seen+=[k]
                    score += entities[split_text[k]]
                    k+=1
                final_entity_list += [conc.strip(" ,.")]
                final_scores += [score/(k-i)]
        return final_entity_list, final_scores
