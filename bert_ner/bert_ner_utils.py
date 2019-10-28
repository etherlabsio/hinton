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

    
    def get_entities(self,text,get_non_entities=False):
        input_ids=[]
        tokenized_text = self.tokenize(text)
        token_to_word=[]
        i=0
        for word in text.split():
            toks = self.tokenizer.encode(word)
            token_to_word.extend([word.lower().strip(".,?!'\"")]*len(toks))
            input_ids.extend(toks)
        # Calculating batch size based on nearest "." from mid-point of text if length exceeds 512
        if len(input_ids)>512:
            mid = len(input_ids)//2
            batch_size = mid - 1 - input_ids[:mid][::-1].index(self.tokenizer.encode(".")[0])
        else:
            batch_size = 510 
        
        for i in range(0,len(input_ids),batch_size):
            encoded_text_sp = self.tokenizer.encode("[CLS]") + input_ids[i:i+batch_size] + self.tokenizer.encode("[SEP]")
            input_ids = torch.tensor(encoded_text_sp).unsqueeze(0)
            entities={}
            non_entities={}
            with torch.no_grad():
                outputs = self.model(input_ids)[0][0,1:-1]
            for i,(tok,embed) in enumerate(zip(token_to_word,list(outputs))):
                embed=embed.unsqueeze(0)
                score = self.sm(embed).detach().numpy().max(-1)[0]
                label = self.labels[self.sm(embed).argmax().detach().numpy()] 
                if label!="O" or (label=="O" and score<0.95):
                    entities[token_to_word[i]] = max(entities.get(token_to_word[i],0),score)
                else:
                    non_entities[tokenized_text[i]] = score
        if get_non_entities:
            return list(entities.keys()),list(entities.values()),list(non_entities.keys()),list(non_entities.values())
        return list(entities.keys()),list(entities.values())
    def tokenize(self,text):
        return self.tokenizer.tokenize(text)

    def wordize(self,tokens, capitalize=False):
        words =  self.tokenizer.convert_tokens_to_string(tokens)
        if capitalize:
            return list(map(lambda x: x.capitalize(),words.split()))
        else:
            return words.upper().split()
