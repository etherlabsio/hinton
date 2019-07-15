from nltk.util import ngrams
import numpy as np
import pandas as pd


#Initialize the object with list of key-phrases and token feature dictionary
class FeatureExtractor(object):

	def __init__(self, embedding_dict, word_weight_dict, stop_words, p_list):
		
		self.embedding_dict = embedding_dict
		self.word_weights = word_weight_dict
		self.p_list = p_list
		self.stop_words = stop_words

	def getTextFeatList(self, text_list):

		text_feat_list = []
		for txt in text_list:
			text_feats = self.getTextFeats(txt)
			text_feat_list.append(text_feats)

		return text_feat_list

	def getTextFeats(self, text):

		p_ctr = len(self.p_list)
		chat = self.getBigramSentence(text.lower(),self.embedding_dict)

		vec_list = []
		feature_vector = []
		extp_feature_vector = []
		word_ctr = 0
	    
		for ele in chat:
		    if ele not in self.stop_words:
		        if ele in self.embedding_dict.keys():
		            word_ctr += 1
		            f_vector = self.embedding_dict[ele]
		            vec_list.append(np.array(f_vector))
		if len(vec_list)>0:
		    for p in self.p_list:
		        t_features = self.getPMeanFeatures(vec_list, p)
		        feature_vector += list(t_features)
		    extp_feature_vector.append(np.array(feature_vector))
		    return extp_feature_vector[0]
		else:
		    return [0]*300*p_ctr

	def getPMeanFeatures(self,vec_list, p):

		if p == 1:
			return np.mean(np.array(vec_list), axis=0)
		else:
		    p_pow_list = []
		    for embed in vec_list:
		        embed = embed ** p
		        p_pow_list.append(embed)
		    p_mat = np.array(p_pow_list)
		    p_mat = np.mean(p_mat, axis=0)
		    p_mat_ = np.array([self.ownpow(item, 1 / p) for item in p_mat])
		    return p_mat_

	def getBigramSentence(self,sent,embedding_dict):

		if len(sent.split(' '))>1:
		    sent = sent.split(' ')
		    sent_merged = ' '.join(sent)
		    t_bigrams = list(ngrams(sent,2))
		    t_bigrams_token = list(pd.Series(t_bigrams).apply(lambda x: x[0]+'_'+x[1]))
		    lookup_tokens = list(set(embedding_dict.keys()).intersection(set(t_bigrams_token)))
		    for bigram_token in lookup_tokens:
		        repl_str = bigram_token.split('_')[0]+' '+bigram_token.split('_')[1]
		        sent_merged = sent_merged.replace(repl_str,bigram_token)
		    split_ = sent_merged.lower().split(' ')
		else:
		    split_ = [sent.lower()]
		return split_

	def ownpow(self, a, b):
		if a > 0:
		    return a ** b
		if a <= 0:
		    temp = abs(a) ** b
		    return -1 * temp

