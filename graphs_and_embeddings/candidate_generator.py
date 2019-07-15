import json
import itertools, nltk, string
import re
from nltk.util import ngrams
import numpy as np
from nltk.tokenize import word_tokenize

class CandidateKPExtractor(object):
	
	def __init__(self, text, stop_words, filter_small_sents = True):

		self.text = text
		self.punct = set(string.punctuation)
		self.filter_small_sents = filter_small_sents
		self.stop_words = stop_words

	def getCandidatePhrases(self, pos_search_pattern_list=[r"""base: {(<JJ.*>*<NN.*>+<IN>)?<JJ>*<NN.*>+}""",
                                        r"""nounverb:{<NN.+>+<.+>{0,2}<VBG>{1}}""",
                                        r"""verbnoun:{<VBG>{1}<.+>{0,2}<NN.+>+}""",
                                       r""" nounnoun:{<NN.+>+<.+>{0,2}<NN.+>+}"""]):
		all_chunks = []
		
		for pattern in pos_search_pattern_list:
			all_chunks+=self.getregexChunks(pattern)

		candidates_tokens = [' '.join(word for word, pos, 
					chunk in group).lower() 
                  	for key, group in itertools.groupby(all_chunks, 
                  	self.lambda_unpack(lambda word, pos, chunk: chunk != 'O')) if key]

		candidate_phrases = [cand for cand in candidates_tokens if cand not in self.stop_words and not all(char in self.punct for char in cand)]
		
		return candidate_phrases

	def getregexChunks(self,grammar):

		chunker = nltk.chunk.regexp.RegexpParser(grammar)
		tagged_sents = nltk.pos_tag_sents(nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(self.text))
		all_chunks = list(itertools.chain.from_iterable(nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
		                                                for tagged_sent in tagged_sents))
		return all_chunks

	def lambda_unpack(self, f):
		return lambda args: f(*args)