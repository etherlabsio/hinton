import nltk
from utils import *
import numpy as np

class PreProcessText(object):

	def __init__(self, text):
		self.text = text

	def getFilteredTextSegment(self):

		tra_sent_list = []
		unfiltered_sent_list = nltk.sent_tokenize(self.text)
		

		for sent_ in unfiltered_sent_list:
			sent_ = stripSpaces(sent_)
			if len(sent_.split(' '))>5:
				tra_sent_list.append(sent_)

		return ' '.join(tra_sent_list),tra_sent_list


def getKPWeight(kp_embedding_dict, sent_embedding_dict):

	score_list = []

	for candidate,candidate_feats in zip(kp_embedding_dict.keys(),kp_embedding_dict.values()):
		curr_scores = []
		for sent_, sent_feats in zip(sent_embedding_dict.keys(),sent_embedding_dict.values()):
			if candidate.lower() in sent_.lower():
				if validVectorCheck(candidate_feats) and validVectorCheck(sent_feats):
					curr_dist = CosineSim(candidate_feats,sent_feats)
				else:
					curr_dist = 0.001
				curr_scores.append(curr_dist)

		score_list.append(np.mean(np.array(curr_scores)))

	kp_weight_dict = dict(zip(kp_embedding_dict.keys(),score_list))

	return kp_weight_dict

