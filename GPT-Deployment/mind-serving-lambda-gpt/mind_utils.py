import numpy as np
import torch
import torch.nn as nn
from gpt.modeling_openai import OpenAIGPTPreTrainedModel,OpenAIGPTConfig,OpenAIGPTModel, SequenceSummary
from gpt.tokenization_openai import OpenAIGPTTokenizer
import io
import zlib
import json
import re
import pickle
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
def load_vocab_files():
    '''
    Returns vocab.json path, merges.txt path and sentence tokenizer object
    '''
    bucket = os.getenv('BUCKET_NAME', 'io.etherlabs.gpt.artifacts')
    vocab_path = os.getenv('VOCAB')
    merges_path = os.getenv('MERGES')
    tokenizer_path = os.getenv('TOKENIZER')

    vocab_dl_path = os.path.join(os.sep, 'tmp', 'vocab.json')
    s3.Bucket(bucket).download_file(vocab_path,vocab_dl_path)

    merges_dl_path = os.path.join(os.sep, 'tmp', 'merges.txt')
    s3.Bucket(bucket).download_file(merges_path,merges_dl_path)

    tokenizer_dl_path = os.path.join(os.sep, 'tmp', 'english.pickle')
    s3.Bucket(bucket).download_file(tokenizer_path,tokenizer_dl_path)

    sent_tokenizer = pickle.load(open(tokenizer_dl_path,"rb"))
    return vocab_dl_path, merges_dl_path, sent_tokenizer

class CustomOpenAIGPTDoubleHeadsModel(OpenAIGPTPreTrainedModel):

    def __init__(self, config):
        super(CustomOpenAIGPTDoubleHeadsModel, self).__init__(config)
        self.transformer = OpenAIGPTModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)
        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.lm_head,
                                   self.transformer.tokens_embed)

    def forward(self, input_ids, mc_token_ids=None, lm_labels=None, mc_labels=None, token_type_ids=None,position_ids=None, head_mask=None):
        
        transformer_outputs = self.transformer(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,head_mask=head_mask)
        hidden_states = transformer_outputs[0]
        return hidden_states[:,:,-1]


special_tokens = ['_start_', '_delimiter_', '_classify_']
vocab_filepath, merges_filepath, sent_tokenizer = load_vocab_files()
tokenizer = OpenAIGPTTokenizer(vocab_filepath,merges_filepath)
tokenizer.add_tokens(special_tokens)
special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)

def splitText(text):
    # returns list of sentences
    if len(text)==0:
        return []
    text = text.strip()
    if not text.endswith((".","?")):
        text+="."
    text = text.replace("?.","?")
    split_text = sent_tokenizer.tokenize(text)
    return split_text

def getReshapedFeatures(lm_tensor):
    lm_array = lm_tensor.reshape(lm_tensor.shape[-1]).detach().numpy()
    return lm_array
    
def getGPTFeatures(model,text):
    max_length = model.config.n_positions-2
        
    encoded_data = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

    input_ids = [special_tokens_ids[0]] + encoded_data[:max_length] + [special_tokens_ids[2]]

    input_tensor = torch.tensor([[input_ids]])

    with torch.no_grad():
        lm_text_feature = model(input_tensor)
    lm_text_feature = getReshapedFeatures(lm_text_feature)
    return lm_text_feature


def getSentenceFeatures(model, mind_dict, input_text):
  #split text into sentences and return sentence feature vector list
    sent_feat_list = []
    sent_list = []
    # punctuate the sentences
    split_text = splitText(input_text)
    for sent in split_text:
        if len(sent)>0:
            sent_feats = getGPTFeatures(model, sent)
            print("In loop ",sent_feats.shape)
            sent_feat_list.append(sent_feats)
            sent_list.append(sent)

    if len(sent_feat_list)>0:
        sent_feat_list = np.array(sent_feat_list)
    feats = list(mind_dict['feature_vector'].values())
    mind_feats_nparray = np.array(feats).reshape(len(feats),-1)
    json_out = {'sent_feats': [sent_feat_list],
                          'mind_feats': [mind_feats_nparray]}
    return json_out

def getParagraphFeatures(model, mind_dict, input_text):
  #split text into sentences and return sentence feature vector list
    sent_feat_list = []
    sent_list = []
    # punctuate the sentences
    split_text = splitText(input_text)
    for sent in split_text:
        if len(sent)>0:
            sent_feats = getGPTFeatures(model, sent)
            sent_feat_list.append(sent_feats)
            sent_list.append(sent)

    if len(sent_feat_list)>0:
        sent_feat_list = np.mean(np.array(sent_feat_list),0).reshape(1,-1)
    feats = list(mind_dict['feature_vector'].values())
    mind_feats_nparray = np.array(feats).reshape(len(feats),-1)
    json_out = {'sent_feats': [sent_feat_list],
                          'mind_feats': [mind_feats_nparray]}
    return json_out