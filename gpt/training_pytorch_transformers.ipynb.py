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

import argparse
import os
import csv
import random
import logging
from tqdm import tqdm, trange, tqdm_notebook as tqnb

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_transformers import (OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer,
                                     AdamW, cached_path, WEIGHTS_NAME, CONFIG_NAME,
                                     WarmupLinearSchedule)

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
""
def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def pre_process_datasets_binclass(encoded_datasets, input_len, cap_length, start_token, delimiter_token, clf_token):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    print("clf_token",clf_token)
    tensor_datasets = []
    for dataset in encoded_datasets:
        #print(dataset)
        n_batch = len(dataset)*2
        input_ids = np.zeros((n_batch, 1, input_len), dtype=np.int64)
        mc_token_ids = np.zeros((n_batch, 1), dtype=np.int64)
        lm_labels = np.full((n_batch, 1, input_len), fill_value=-1, dtype=np.int64)
        mc_labels = np.zeros((n_batch), dtype=np.int64)
        i=0
        for story, cont1, cont2, mc_label in dataset:
            with_cont1 = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
            with_cont2 = [start_token] + story[:cap_length] + [delimiter_token] + cont2[:cap_length] + [clf_token]
            input_ids[i, 0, :len(with_cont1)] = with_cont1
            input_ids[i+1, 0, :len(with_cont2)] = with_cont2
            mc_token_ids[i, 0] = len(with_cont1) - 1
            mc_token_ids[i+1, 0] = len(with_cont2) - 1
            lm_labels[i, 0, :len(with_cont1)] = with_cont1
            lm_labels[i+1, 0, :len(with_cont2)] = with_cont2
            if mc_label==0:
                mc_labels[i] = 1
                mc_labels[i+1] = 0
            else:
                mc_labels[i] = 0
                mc_labels[i+1] = 1
            i+=2
        all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)
        tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
    return tensor_datasets

def pre_process_datasets(encoded_datasets, input_len, cap_length, start_token, delimiter_token, clf_token):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    print("clf_token",clf_token)
    tensor_datasets = []
    for dataset in encoded_datasets:
        #print(dataset)
        n_batch = len(dataset)
        input_ids = np.zeros((n_batch, 2, input_len), dtype=np.int64)
        mc_token_ids = np.zeros((n_batch, 2), dtype=np.int64)
        lm_labels = np.full((n_batch, 2, input_len), fill_value=-1, dtype=np.int64)
        mc_labels = np.zeros((n_batch), dtype=np.int64)
        for i, (story, cont1, cont2, mc_label), in enumerate(dataset):
            with_cont1 = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
            with_cont2 = [start_token] + story[:cap_length] + [delimiter_token] + cont2[:cap_length] + [clf_token]
            input_ids[i, 0, :len(with_cont1)] = with_cont1
            input_ids[i, 1, :len(with_cont2)] = with_cont2
            mc_token_ids[i, 0] = len(with_cont1) - 1
            mc_token_ids[i, 1] = len(with_cont2) - 1
            lm_labels[i, 0, :len(with_cont1)] = with_cont1
            lm_labels[i, 1, :len(with_cont2)] = with_cont2
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
            output.append(('.'.join(line[0:-3]), line[-3], line[-2], int(line[-1])))
    return output

def tokenize_and_encode(obj):
    """ Tokenize and encode a nested object """
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    elif isinstance(obj, int):
        return obj
    return list(tokenize_and_encode(o) for o in obj)

# +
# Defining constants over here
seed = 42 
# model_name = '/home/ether/hdd/ether/gpt_domain_minds/se/epoch3/'
model_name = 'openai-gpt'

train_dataset = '../data/customer_service_ROC_data.csv'
# train_dataset = "../data/data_original.csv"
# train_dataset = "../data/se_data/se_nsp_4+w_data.csv"

eval_dataset = train_dataset
do_train = True
output_dir = '/home/ether/hdd/arjun/models_bkp/gpt_models/cs_model/'
save_model_every_epoch = True
parallel_train = True
num_train_epochs = 3
train_batch_size = 4
eval_batch_size = 1
max_grad_norm = 1
learning_rate = 6.25e-5
warmup_proportion = 0.002 
lr_schedule = 'warmup_linear'
weight_decay = 0.01
lm_coef = 0.9
mc_coef = 1
n_valid = 374
max_steps = -1
gradient_accumulation_steps = 1
adam_epsilon = 1e-8
warmup_steps = 0
try:
    os.mkdir(output_dir+"/")
except OSError as e:
    print(e)
    pass
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# -

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logger.info("device: {}, n_gpu {}".format(device, n_gpu))


"""
# Load tokenizer and model
# This loading functions also add new tokens and embeddings called `special tokens`
# These new embeddings will be fine-tuned on the RocStories dataset
special_tokens = ['_start_', '_delimiter_', '_classify_']
tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name, special_tokens=special_tokens)
special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
model = OpenAIGPTDoubleHeadsModel.from_pretrained(model_name, num_special_tokens=len(special_tokens))
model.to(device)
"""
def save_model(output_dir):
    try:
        os.mkdir(output_dir+"/")
    except OSError:
        pass
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


special_tokens = ['_start_', '_delimiter_', '_classify_']
tokenizer = OpenAIGPTTokenizer.from_pretrained(model_name)
tokenizer.add_tokens(special_tokens)
special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
model = OpenAIGPTDoubleHeadsModel.from_pretrained(model_name,num_labels=2)
model.resize_token_embeddings(len(tokenizer))
model.to(device)


logger.info("Encoding dataset...")

train_dataset = load_rocstories_dataset(train_dataset)
print('length dataset',len(train_dataset))
datasets = (train_dataset[:],)
print(len(datasets[0]))
encoded_datasets = tokenize_and_encode(datasets)
# Compute the max input length for the Transformer
max_length = model.config.n_positions // 2 - 2
input_length = max(len(story[:max_length]) + max(len(cont1[:max_length]), len(cont2[:max_length])) + 3  \
                        for dataset in encoded_datasets for story, cont1, cont2, _ in dataset)
input_length = min(input_length, model.config.n_positions)  # Max size of input for the pre-trained model

# Prepare inputs tensors and dataloaders
tensor_datasets = pre_process_datasets(encoded_datasets, input_length, max_length, *special_tokens_ids)
train_tensor_dataset = tensor_datasets[0]
train_data = TensorDataset(*train_tensor_dataset)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)





if parallel_train:
    model = torch.nn.DataParallel(model,output_device=torch.device("cuda:1"))



# Prepare optimizer
if do_train:
    if max_steps > 0:
        t_total = max_steps
        num_train_epochs = max_steps //\
            (len(train_dataloader) // gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader)\
            // gradient_accumulation_steps * num_train_epochs
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total)

if do_train:
    nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
    model.train()
    for ep_no in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_steps = 0
        tqdm_bar = tqdm(train_dataloader, desc="Training")
        for step, batch in enumerate(tqdm_bar):
            batch = tuple(t.to(device) for t in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels = batch
            losses = model(input_ids, mc_token_ids, lm_labels, mc_labels)
            lm_loss = losses[0].mean()
            mc_loss = losses[1].mean()
            loss = (lm_coef * lm_loss) + (mc_coef * mc_loss)
            
            loss.backward()
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            tr_loss += loss.item()
            exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
            nb_tr_steps += 1
            tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, scheduler.get_lr()[0])
            # Save a trained model, configuration and tokenizer every step steps
#             if step%((len(train_dataloader)-2)//3)==0:
#                 save_model(output_dir+"_ep"+str(1+ep_no)+"_step"+str(step))
        # Save a trained model, configuration and tokenizer every epoch
        if save_model_every_epoch:
            save_model(output_dir+"epoch_"+str(1+ep_no))







# Save a trained model
if do_train:
    # Save a trained model, configuration and tokenizer
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    
    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)

    # Load a trained model and vocabulary that you have fine-tuned
    model = OpenAIGPTDoubleHeadsModel.from_pretrained(output_dir)
    tokenizer = OpenAIGPTTokenizer.from_pretrained(output_dir)
    model.to(device)









