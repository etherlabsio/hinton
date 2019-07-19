from pytorch_pretrained_bert import BertTokenizer, BertConfig, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertPreTrainingHeads
import torch
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import time

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


class BertForPreTrainingCustom(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForPreTrainingCustom, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None):
        output_all_encoded_layers = True
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=output_all_encoded_layers)
        if output_all_encoded_layers:
            sequence_output_pred = sequence_output[-1]
        prediction_scores, seq_relationship_score = self.cls(sequence_output_pred, pooled_output)
        return prediction_scores, seq_relationship_score, sequence_output, pooled_output

def getBERTFeatures(model, text, attn_head_idx=-1):  # attn_head_idx - index o[]

    """
    Get BERT features for the `text`
    Args:
        model: BERT model of type `BertForPreTrainingCustom`
        text: required, get features for this text
        attn_head_idx: optional, defaults to last layer
    Returns:
        tuple - {token_feats[attn_head_idx][0],final_feats,tokenized_text}
    """

    tokenized_text = tokenizer.tokenize(text)
    if len(tokenized_text) > 200:
        tokenized_text = tokenized_text[0:200]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    #print('indexed_tokens: ', indexed_tokens)
    tokens_tensor = torch.tensor([indexed_tokens])
    #print('tokens_tensor: ', tokens_tensor)
    _, _, token_feats, pool_out = model(tokens_tensor)
    final_feats = list(getPooledFeatures(token_feats[attn_head_idx]).T)
    return token_feats[attn_head_idx][0],final_feats,tokenized_text

def getPooledFeatures(np_array):
    np_array = np_array.reshape(np_array.shape[1], np_array.shape[2]).detach().numpy()
    np_array_mp = np.mean(np_array, axis=0).reshape(1, -1)
    return np_array_mp