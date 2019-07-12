### Code to extract nsp scores and features
import json
from typing import NamedTuple
#import fire

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import checkpoint
import tokenization
import optim
import trainer
import data
import models
import time

from utils import set_seeds, get_device


class Config(NamedTuple):
    """ Config for classification """
    mode: str = "eval"
    seed: int = 12345
    cfg_data: str = "EtherText.json"
    cfg_model: str = "uncased_L-12_H-768_A-12/bert_base.json"
    cfg_optim: str = "optim.json"
    model_file: str = "/home/shivam/Desktop/bert_distillation/models/ether/distill/model_final.pt"
    pretrain_file: str = "uncased_L-12_H-768_A-12/bert_model.ckpt"
    save_dir: str = "exp"
    comments: str = [] # for comments in json file


def Scorer(sent_1,sent_2):

    instance = "0" ,sent_1,sent_2    #####0 means context  1 means non context
    cfg = Config() #**json.load(open(config, "r"))
    cfg_data = data.Config(**json.load(open(cfg.cfg_data, "r")))
    cfg_model = models.Config(**json.load(open(cfg.cfg_model, "r")))
    cfg_optim = trainer.Config(**json.load(open(cfg.cfg_optim, "r")))
    set_seeds(cfg.seed)

    ### Prepare Dataset and Preprocessing ###
    TaskDataset = data.get_class(cfg_data.task) # task dataset class according to the task
    tokenizer = tokenization.FullTokenizer(vocab_file=cfg_data.vocab_file, do_lower_case=True)

    dataset = TaskDataset(instance, pipelines=[
        data.RemoveSymbols('\\'),
        data.Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
        data.AddSpecialTokensWithTruncation(cfg_data.max_len),
        data.TokenIndexing(tokenizer.convert_tokens_to_ids,
                           TaskDataset.labels,
                           cfg_data.max_len)
    ], n_data=None)
    
    tensors = TensorDataset(*dataset.get_tensors()) # To Tensors
    data_iter = DataLoader(tensors, batch_size=cfg_optim.batch_size, shuffle=False)

    ### Models ###
    model = models.CNN_2(cfg_model, len(TaskDataset.labels))
    checkpoint.load_embedding(model.embed, cfg.pretrain_file)
    optimizer = optim.optim4GPU(cfg_optim, model)

    train_loop = trainer.TrainLoop(
        cfg_optim, model, data_iter, optimizer, cfg.save_dir, get_device()
    )
    
    def extract_nsp_score(model, batch):
        start = time.time()
        input_ids, segment_ids, input_mask, label_id = batch
        
        logits = model(input_ids, segment_ids, input_mask)
        end = time.time()
        nsp_score, label_pred = logits.max(1)
        result = (label_pred == label_id).float() #.cpu().numpy()
        accuracy = result.mean()
        
        print('time = ' + str(end - start))
        return accuracy, result, logits.cpu().numpy()


    if cfg.mode == "eval":
        results = train_loop.nsp_scorer(extract_nsp_score, cfg.model_file)
        print(f"NSP_Score: {results}")


def Feature_extrator(sent_1,sent_2=None):

    instance = "0" ,sent_1,sent_2    #####0 means context  1 means non context
    cfg = Config() #**json.load(open(config, "r"))
    cfg_data = data.Config(**json.load(open(cfg.cfg_data, "r")))
    cfg_model = models.Config(**json.load(open(cfg.cfg_model, "r")))
    cfg_optim = trainer.Config(**json.load(open(cfg.cfg_optim, "r")))
    set_seeds(cfg.seed)

    ### Prepare Dataset and Preprocessing ###
    TaskDataset = data.get_class(cfg_data.task) # task dataset class according to the task
    tokenizer = tokenization.FullTokenizer(vocab_file=cfg_data.vocab_file, do_lower_case=True)

    dataset = TaskDataset(instance, pipelines=[
        data.RemoveSymbols('\\'),
        data.Tokenizing(tokenizer.convert_to_unicode, tokenizer.tokenize),
        data.AddSpecialTokensWithTruncation(cfg_data.max_len),
        data.TokenIndexing(tokenizer.convert_tokens_to_ids,
                           TaskDataset.labels,
                           cfg_data.max_len)
    ], n_data=None)
    
    tensors = TensorDataset(*dataset.get_tensors()) # To Tensors
    data_iter = DataLoader(tensors, batch_size=cfg_optim.batch_size, shuffle=False)

    ### Models ###
    features = models.Features_Blend_CNN(cfg_model, len(TaskDataset.labels))
    checkpoint.load_embedding(features.embed, cfg.pretrain_file)
    optimizer = optim.optim4GPU(cfg_optim, features)

    train_loop = trainer.TrainLoop(
        cfg_optim, features, data_iter, optimizer, cfg.save_dir, get_device()
    )

    def extract_features(model, batch):
        #start = time.time()
        input_ids, segment_ids, input_mask, _ = batch
        vector = features(input_ids, segment_ids, input_mask)
        #end = time.time()        
        #print('time = ' + str(end - start))
        return vector

    if cfg.mode == "eval":
        feature_vector = train_loop.feature_extraction(extract_features, cfg.model_file)
        print(f"Feature_vector: {feature_vector}")
        


if __name__ == '__main__':
    sent_1 = 'WebRTC (Web Real-Time Communication) is a free, open-source project'
    sent_2 = "Webrtc provides web browsers and mobile applications with real-time communication (RTC)"
    Scorer(sent_1,sent_2)
    Feature_extrator(sent_1)
