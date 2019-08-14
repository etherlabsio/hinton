import argparse
import os
import csv
import random
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
							  TensorDataset)
from pytorch_pretrained_bert import (OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
									 OpenAIAdam, cached_path, WEIGHTS_NAME, CONFIG_NAME)
from pytorch_pretrained_bert.modeling_openai import OpenAIGPTPreTrainedModel,OpenAIGPTDoubleHeadsModel,OpenAIGPTConfig,OpenAIGPTModel,OpenAIGPTLMHead

from scipy.spatial.distance import cosine

##############################################################################

# Defining constants over here
seed = 42 
model_name = 'openai-gpt'
#train_dataset = '/home/shubham/Project/domain_mind/gpt2_experiment/data/data_original.csv'
#eval_dataset = '/home/shubham/Project/domain_mind/gpt2_experiment/data/data_original.csv'
#do_train = True
#output_dir = './model/'
#output_dir = './model2'
#num_train_epochs = 1
train_batch_size = 64
#eval_batch_size = 16
#max_grad_norm = 1
#learning_rate = 6.25e-5
#warmup_proportion = 0.002 
#lr_schedule = 'warmup_linear'
#weight_decay = 0.01
#lm_coef = 0.9
#n_valid = 374

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()

#output_dir = './model2'


###############################################################################

class OpenAIGPTLMHead_custom(nn.Module):
	""" Language Model Head for the transformer """

	def __init__(self, model_embeddings_weights, config):
		super(OpenAIGPTLMHead_custom, self).__init__()
		self.n_embd = config.n_embd
		self.vocab_size = config.vocab_size
		self.predict_special_tokens = config.predict_special_tokens
		embed_shape = model_embeddings_weights.shape
		#print("shape check",(model_embeddings_weights[1]))
		self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
		self.set_embeddings_weights(model_embeddings_weights)

	def set_embeddings_weights(self, model_embeddings_weights, predict_special_tokens=True):
		self.predict_special_tokens = predict_special_tokens
		embed_shape = model_embeddings_weights.shape
		self.decoder.weight = model_embeddings_weights  # Tied weights

	def forward(self, hidden_state):
		#print('decoder weight')
		#print((hidden_state.shape))
		lm_logits = self.decoder(hidden_state)
		#print(lm_logits.shape)
		if not self.predict_special_tokens:
			lm_logits = lm_logits[..., :self.vocab_size]
			#print("lm_logits.shape: ",lm_logits.shape)
		return lm_logits

class OpenAIGPTMultipleChoiceHead_custom(nn.Module):
	""" Classifier Head for the transformer """

	def __init__(self, config):
		super(OpenAIGPTMultipleChoiceHead_custom, self).__init__()
		self.n_embd = config.n_embd
		self.dropout = nn.Dropout2d(config.resid_pdrop)  # To reproduce the noise_shape parameter of TF implementation
		self.linear = nn.Linear(config.n_embd, 1)

		nn.init.normal_(self.linear.weight, std=0.02)
		nn.init.normal_(self.linear.bias, 0)

	def forward(self, hidden_states, mc_token_ids):
		# Classification logits
		# hidden_state (bsz, num_choices, seq_length, hidden_size)
		# mc_token_ids (bsz, num_choices)
		mc_token_ids = mc_token_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, hidden_states.size(-1))
		# (bsz, num_choices, 1, hidden_size)
		#print('mc_token_ids: ', mc_token_ids[0][0].shape,mc_token_ids[0][1].shape)
		#print('mc_token_ids.shape: ', mc_token_ids.shape)
		#print('Hidden states before compute: ', hidden_states.shape)
		multiple_choice_h = hidden_states.gather(2, mc_token_ids).squeeze(2)
		#print('After transformation: ', multiple_choice_h.shape)
		# (bsz, num_choices, hidden_size)
		#multiple_choice_h = self.dropout(multiple_choice_h.transpose(1, 2)).transpose(1, 2)
		#multiple_choice_logits = self.linear(multiple_choice_h).squeeze(-1)
		# (bsz, num_choices)
		return multiple_choice_h

class OpenAIGPTDoubleHeadsModel_custom(OpenAIGPTPreTrainedModel):
	"""
	OpenAI GPT model with a Language Modeling and a Multiple Choice head ("Improving Language Understanding by Generative Pre-Training").
	OpenAI GPT use a single embedding matrix to store the word and special embeddings.
	Special tokens embeddings are additional tokens that are not pre-trained: [SEP], [CLS]...
	Special tokens need to be trained during the fine-tuning if you use them.
	The number of special embeddings can be controled using the `set_num_special_tokens(num_special_tokens)` function.
	The embeddings are ordered as follow in the token embeddings matrice:
		[0,                                                         ----------------------
		 ...                                                        -> word embeddings
		 config.vocab_size - 1,                                     ______________________
		 config.vocab_size,
		 ...                                                        -> special embeddings
		 config.vocab_size + config.n_special - 1]                  ______________________
	where total_tokens_embeddings can be obtained as config.total_tokens_embeddings and is:
		total_tokens_embeddings = config.vocab_size + config.n_special
	You should use the associate indices to index the embeddings.
	Params:
		`config`: a OpenAIGPTConfig class instance with the configuration to build a new model
		`output_attentions`: If True, also output attentions weights computed by the model at each layer. Default: False
		`keep_multihead_output`: If True, saves output of the multi-head attention module with its gradient.
			This can be used to compute head importance metrics. Default: False
	Inputs:
		`input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length] with the BPE token
			indices selected in the range [0, total_tokens_embeddings[
		`mc_token_ids`: a torch.LongTensor of shape [batch_size, num_choices] with the index of the token from
			which we should take the hidden state to feed the multiple choice classifier (usually last token of the sequence)
		`position_ids`: an optional torch.LongTensor with the same shape as input_ids
			with the position indices (selected in the range [0, config.n_positions - 1[.
		`token_type_ids`: an optional torch.LongTensor with the same shape as input_ids
			You can use it to add a third type of embedding to each input token in the sequence
			(the previous two being the word and position embeddings).
			The input, position and token_type embeddings are summed inside the Transformer before the first
			self-attention block.
		`lm_labels`: optional language modeling labels: torch.LongTensor of shape [batch_size, num_choices, sequence_length]
			with indices selected in [-1, 0, ..., total_tokens_embeddings]. All labels set to -1 are ignored (masked), the loss
			is only computed for the labels set in [0, ..., total_tokens_embeddings]
		`multiple_choice_labels`: optional multiple choice labels: torch.LongTensor of shape [batch_size]
			with indices selected in [0, ..., num_choices].
		`head_mask`: an optional torch.Tensor of shape [num_heads] or [num_layers, num_heads] with indices between 0 and 1.
			It's a mask to be used to nullify some heads of the transformer. 1.0 => head is fully masked, 0.0 => head is not masked.
	Outputs:
		if `lm_labels` and `multiple_choice_labels` are not `None`:
			Outputs a tuple of losses with the language modeling loss and the multiple choice loss.
		else: a tuple with
			`lm_logits`: the language modeling logits as a torch.FloatTensor of size [batch_size, num_choices, sequence_length, total_tokens_embeddings]
			`multiple_choice_logits`: the multiple choice logits as a torch.FloatTensor of size [batch_size, num_choices]
	Example usage:
	```python
	# Already been converted into BPE token ids
	input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]]])  # (bsz, number of choice, seq length)
	mc_token_ids = torch.LongTensor([[2], [1]]) # (bsz, number of choice)
	config = modeling_openai.OpenAIGPTOpenAIGPTMultipleChoiceHead_customOpenAIGPTMultipleChoiceHead_customConfig()
	model = modeling_openai.OpenAIGPTDoubleHeadsModel(config)
	lm_logits, multiple_choice_logits = model(input_ids, mc_token_ids)
	```
	"""

	def __init__(self, config, output_attentions=False, keep_multihead_output=False):
		super(OpenAIGPTDoubleHeadsModel_custom, self).__init__(config)
		self.transformer = OpenAIGPTModel(config, output_attentions=False,
											 keep_multihead_output=keep_multihead_output)
		self.lm_head = OpenAIGPTLMHead_custom(self.transformer.tokens_embed.weight, config)
		self.multiple_choice_head = OpenAIGPTMultipleChoiceHead_custom(config)
		self.apply(self.init_weights)

	def set_num_special_tokens(self, num_special_tokens, predict_special_tokens=True):
		""" Update input and output embeddings with new embedding matrice
			Make sure we are sharing the embeddings
		"""
		#self.config.predict_special_tokens = self.transformer.config.predict_special_tokens = predict_special_tokens
		self.transformer.set_num_special_tokens(num_special_tokens)
		self.lm_head.set_embeddings_weights(self.transformer.tokens_embed.weight, predict_special_tokens=predict_special_tokens)

	def forward(self, input_ids, mc_token_ids, lm_labels=None, mc_labels=None, token_type_ids=None,
				position_ids=None, head_mask=None):
		hidden_states = self.transformer(input_ids, position_ids, token_type_ids, head_mask)
		if self.transformer.output_attentions:
			all_attentions, hidden_states = hidden_states
		#print('hidden states',len(hidden_states))
		hidden_states = hidden_states[-1]
		#return hidden_states[0][0][-2],hidden_states[0][0][-2],hidden_states[0][0][-2]
		lm_logits = self.lm_head(hidden_states)
		hidden_feats = self.multiple_choice_head(hidden_states, mc_token_ids)
		losses = []
		if lm_labels is not None:
			shift_logits = lm_logits[..., :-1, :].contiguous()
			shift_labels = lm_labels[..., 1:].contiguous()
			loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
			losses.append(loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)))
		return lm_logits, hidden_feats,hidden_states[0][0][-2]

###############################################################################  

def accuracy(out, labels):
	outputs = np.argmax(out, axis=1)
	return np.sum(outputs == labels)

def pre_process_datasets(encoded_datasets, input_len, cap_length, start_token, delimiter_token, clf_token):
	""" Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

		To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
		input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
	"""
	#print("clf_token",delimiter_token)
	tensor_datasets = []
	for dataset in encoded_datasets:
		#print(dataset)
		n_batch = len(dataset)
		input_ids = np.zeros((n_batch, 1, input_len), dtype=np.int64)
		mc_token_ids = np.zeros((n_batch, 1), dtype=np.int64)
		lm_labels = np.full((n_batch, 1, input_len), fill_value=-1, dtype=np.int64)
		mc_labels = np.zeros((n_batch,), dtype=np.int64)
		for i, (story, cont1, cont2, mc_label), in enumerate(dataset):
			#with_cont1 = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
			with_cont1 = [start_token] + story[:cap_length] + [clf_token]
			#print(len(with_cont1))
			#with_cont2 = [start_token] + story[:cap_length] + [delimiter_token] + cont2[:cap_length] + [clf_token]
			#with_cont2 = [start_token] + cont1[:cap_length] + [clf_token]
			input_ids[i, 0, :len(with_cont1)] = with_cont1
			#input_ids[i, 1, :len(with_cont2)] = with_cont2
			mc_token_ids[i, 0] = len(with_cont1) - 1
			#mc_token_ids[i, 1] = len(with_cont2) - 1
			lm_labels[i, 0, :len(with_cont1)] = with_cont1
			#lm_labels[i, 1, :len(with_cont2)] = with_cont2
			mc_labels[i] = mc_label
		all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)
		tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
	return tensor_datasets

def load_rocstories_dataset(dataset_path):
	""" Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
	with open(dataset_path, encoding='utf_8') as f:
		f = csv.reader(f)
		output = []
		next(f) # skip the first line
		for line in tqdm(f):
			output.append(('.'.join(line[0 :4]), line[4], line[5], int(line[-1])))
	return output

def tokenize_and_encode(obj,tokenizer):
	""" Tokenize and encode a nested object """
	if isinstance(obj, str):
		return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
	elif isinstance(obj, int):
		return obj
	return list(tokenize_and_encode(o,tokenizer) for o in obj)


'''
special_tokens = ['_start_', '_delimiter_', '_classify_']
tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name, special_tokens=special_tokens)
special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)

model1 = OpenAIGPTDoubleHeadsModel_custom.from_pretrained(output_dir)
tokenizer = OpenAIGPTTokenizer.from_pretrained(output_dir)
model1.to(device)
model1.eval()
'''

def feature_extractor(model1,text):
	#train_dataset = load_rocstories_dataset(train_dataset)
	#print(len(train_dataset[1:2]))
	#eval_dataset = load_rocstories_dataset(eval_dataset)
	#print(eval_dataset[0])
	special_tokens = ['_start_', '_delimiter_', '_classify_']
	tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name, special_tokens=special_tokens)
	special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
	trn_dt = ([text,'','',0],)   
	#datasets = (train_dataset[1:2],)
	datasets = (trn_dt,)
	#print(datasets)
	encoded_datasets = tokenize_and_encode(datasets,tokenizer)
	# Compute the max input length for the Transformer
	max_length = model1.config.n_positions // 2 - 2
	input_length = max(len(story[:max_length]) + max(len(cont1[:max_length]), len(cont2[:max_length])) + 3  \
							for dataset in encoded_datasets for story, cont1, cont2, _ in dataset)
	input_length = min(input_length, model1.config.n_positions)  # Max size of input for the pre-trained model
	# Prepare inputs tensors and dataloaders
	tensor_datasets = pre_process_datasets(encoded_datasets, input_length, max_length, *special_tokens_ids)
	train_tensor_dataset = tensor_datasets[0]
	train_data = TensorDataset(*train_tensor_dataset)
	train_sampler = RandomSampler(train_data)
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)
	
	for batch in train_dataloader:
		batch = tuple(t.to(device) for t in batch)
		input_ids, mc_token_ids, lm_labels, mc_labels = batch
		with torch.no_grad():
			a, clf_text_feature,lm_text_feature = model1(input_ids, mc_token_ids, lm_labels, mc_labels)
			#print('mc_loss',mc_loss[0][1].shape)
				
	return clf_text_feature	, lm_text_feature


'''
test_text1 = 'Docker is a set of coupled software-as-a-service and platform-as-a-service products that use operating-system-level virtualization to develop and deliver software in packages called containers.'
test_text2 = 'HR managers with legal information for both state and federal law.'	
#test_text1 = 'SQL stands for Structured Query Language. It is designed for managing data in a relational database management system (RDBMS).'
test_text2 = 'Kubernetes is an open-source container-orchestration system for automating application deployment, scaling, and management. It was originally designed by Google, and is now maintained by the Cloud Native Computing Foundation'
#test_text2 = 'In project management, products are the formal definition of the project deliverables that make up or contribute to delivering the objectives of the project.'
t1clf , t1lm = feature_extractor(model1,test_text1)
t2clf , t2lm = feature_extractor(model1,test_text2)

cosine_distance = 1-cosine(t1clf, t2clf)
print('Cosine Similarity clf: ', 1-cosine_distance)

cosine_distance1 = 1-cosine(t1lm, t2lm)
print('Cosine Similarity lm: ', 1-cosine_distance1)
'''
