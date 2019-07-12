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

def getClusterMeanDistance(transcript_feature_list, topic_embedding, cluster_embed):
    # returns the cosine similarity score between this transcript and current topic cluster
    tra_dist_list = []  # return min of this

    for curr_sent in transcript_feature_list:
        sent_topic_dists = []
        for topic_feats in topic_embedding:
            sent_topic_dists.append(cosine(curr_sent, topic_feats))

        # sent_topic_dist = np.mean(sent_topic_dists)
        sent_clust_dist = cosine(curr_sent, cluster_embed)

        # sent_dist = np.mean([sent_topic_dist,sent_clust_dist])
        # sent_dist = sent_clust_dist
        tra_dist_list.append(sent_clust_dist)

    return np.min(tra_dist_list)  # min_pool

def getBERTFeatDF(tran_dg, cluster_centers, model, dist_threshold):
    filtered_transcripts = []
    filtered_rec_ids = []
    transcript_feature_list = []

    ctr = 0
    for transcript, rec_id in zip(list(tran_dg['filteredText']), list(tran_dg['recordingId'])):
        transcript_feats = []
        if len(transcript.split(' ')) > 10:
            filtered_transcripts.append(transcript)
            filtered_rec_ids.append(rec_id)
            for sent in transcript.split('.'):
                if len(sent.strip().split(' ')) > 3:
                    transcript_feats.append(getBERTFeatures(model, sent.strip()))
            transcript_feature_list.append(transcript_feats)

    master_dist_list = []
    for curr_tra_feats in transcript_feature_list:
        curr_seg_dists = []
        for sent_feats in curr_tra_feats:
            sent_dist_list = []
            for centroid in cluster_centers:
                sent_dist_list.append(cosine(centroid, sent_feats))
            curr_seg_dists.append(sent_dist_list)
        master_dist_list.append(curr_seg_dists)

    # rank transcript sentences by their minimum score in atlease n that pass the threshold
    transcript_candidate_ctr = []
    transcript_dist_score = []
    transcript_rel = []
    transcript_length_list = []

    for transcript, dist_list in zip(filtered_transcripts, master_dist_list):
        candidate_ctr = 0
        tra_mean_dist = []
        for sent_dist in dist_list:
            # take sentences in this transcripts with < threshold
            candidate_idxs = list(np.ndarray.flatten(np.argwhere(np.array(sent_dist) < dist_threshold)))
            # print('candidate_idxs: ', candidate_idxs)
            # take minimum of these idx values
            if len(candidate_idxs) > 1 and len(candidate_idxs) < 4:  # relavant to atleast two and not more than 3
                # if len(candidate_idxs)>1:
                sent_min_dist = np.min([sent_dist[i] for i in candidate_idxs])
                tra_mean_dist.append(sent_min_dist)
                candidate_ctr += 1

        transcript_length_list.append(len(dist_list))
        transcript_rel.append(candidate_ctr / len(dist_list))
        transcript_candidate_ctr.append(candidate_ctr)
        transcript_dist_score.append(np.mean(tra_mean_dist))

    merged_df = pd.DataFrame({'Transcript': filtered_transcripts,
                              'Recording_ID': filtered_rec_ids,
                              'Sent_Counter': transcript_candidate_ctr,
                              'Rel_Score': transcript_rel,
                              'Transcript_length': transcript_length_list,
                              'Dist_Score': transcript_dist_score})

    merged_df = merged_df.sort_values(by=['Sent_Counter', 'Dist_Score'], ascending=[False, True]).reset_index(drop=True)
    merged_df['Dist_Score'] = merged_df['Dist_Score'].fillna(10000)
    merged_df['metric'] = merged_df['Sent_Counter'] / merged_df['Dist_Score']
    merged_df['metric'] = merged_df['metric'].apply(lambda x: x if x > 0 else 0.001)
    merged_df['inv'] = 1 / merged_df['metric']

    return merged_df

def getNSPScore(sample_text, model):
    m = torch.nn.Softmax()

    tokenized_text = tokenizer.tokenize(sample_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [0] * tokenized_text.index('[SEP]') + [1] * (len(tokenized_text) - tokenized_text.index('[SEP]'))

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    pred_score, seq_rel, seq_out, pool_out = model(tokens_tensor, segments_tensors)
    return m(seq_rel).detach().numpy()[0][0]  # returns probability of being next sentence

def getRelMetric(sent1, sent2, model, attn_head_idx=-1, nsp_dampening_factor=0.7):

    tot_start = time.time()
    sent1_feats = getBERTFeatures(model, sent1.strip(), attn_head_idx)
    sent2_feats = getBERTFeatures(model, sent2.strip(), attn_head_idx)
    cosine_sim = 1 - cosine(sent1_feats, sent2_feats)

    nsp_input1 = sent1 + ' [SEP] ' + sent2
    nsp_input2 = sent2 + ' [SEP] ' + sent1

    nsp_score_1 = getNSPScore(nsp_input1, model)

    #nsp_score_2 = getNSPScore(nsp_input2, model)
    #nsp_score = np.mean([nsp_score_1, nsp_score_2]) * nsp_dampening_factor
    nsp_score = nsp_score_1*nsp_dampening_factor

    len_diff = abs(len(sent1.split(' ')) - len(sent2.split(' ')))
    if len_diff > 2 * (min(len(sent1.split(' ')), len(sent2.split(' ')))):
        score = 0.4 * cosine_sim + 0.6 * nsp_score
    else:
        score = np.mean([cosine_sim, nsp_score])
    tot_end = time.time()
    #print('')
    #print('Sent time:',tot_end-tot_start)
    return score

def getSentMatchScore(sent1, sent2, model, attn_head_idx,nsp_dampening_factor = 0.7):
    
    sent1_feats = getBERTFeatures(model, sent1.strip(), attn_head_idx)
    sent2_feats = getBERTFeatures(model, sent2.strip(), attn_head_idx)
    
    cosine_sim = 1-cosine(sent1_feats, sent2_feats)
    
    nsp_input1 = sent1+' [SEP] '+sent2
    nsp_input2 = sent2+' [SEP] '+sent1
    
    nsp_score_1 = getNSPScore(nsp_input1, model)
    #nsp_score = np.mean([nsp_score_1,nsp_score_2])*nsp_dampening_factor
    nsp_score = nsp_score_1*nsp_dampening_factor
    
#     len_diff = abs(len(sent1.split(' '))-len(sent2.split(' ')))
#     if len_diff>2*(min(len(sent1.split(' ')),len(sent2.split(' ')))):
#         score = 0.4*cosine_sim+0.6*nsp_score
#     else:
    score = np.mean([cosine_sim,nsp_score])
        
    return score