# -*- coding:utf-8 -*-
"Evaluate the model"""
import os
import nltk
import torch
import random
import logging
import argparse
import numpy as np
import utils as utils
from metrics import get_entities
from data_loader import DataLoader
from SequenceTagger import BertForSequenceTagging
from transformers import AutoModel
from flask import Flask, request
import json
import time

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='msra',
                    help="Directory containing the dataset")
parser.add_argument('--seed', type=int, default=23,
                    help="random seed for initialization")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def interAct(model, data_iterator, params, mark='Interactive', verbose=False):
    """Evaluate the model on `steps` batches."""
    # set model to evaluation mode
    model.eval()
    idx2tag = params.idx2tag

    batch_data, batch_token_starts = next(data_iterator)
    batch_masks = batch_data.gt(0)

    batch_output = model((batch_data, batch_token_starts), token_type_ids=None,
                         attention_mask=batch_masks)[0]  # shape: (batch_size, max_len, num_labels)
    batch_output = batch_output.detach().cpu().numpy()

    pred_tags = []
    pred_tags.extend([[idx2tag.get(idx) for idx in indices]
                      for indices in np.argmax(batch_output, axis=2)])

    return(get_entities(pred_tags))


def bert_ner_init():
    args = parser.parse_args()
    # tagger_model_dir = 'experiments/based_finbert/' + args.dataset
    tagger_model_dir = './ner_ours_test/experiments/based_finbert'

    # Load the parameters from json file
    json_path = os.path.join(tagger_model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Use GPUs if available
    # params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params.device = torch.device('cpu')

    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    params.seed = args.seed

    # Set the logger
    # utils.set_logger(os.path.join(tagger_model_dir, './ner_ours_test/evaluate.log'))

    # Create the input data pipeline
    logging.info("Loading the dataset...")

    # Initialize the DataLoader
    data_dir = './ner_ours_test/data/' + args.dataset
    bert_class = './ner_ours_test/pretrained_bert_models/FinBERT_L-12_H-768_A-12_pytorch'

    data_loader = DataLoader(data_dir, bert_class,
                             params, token_pad_idx=0, tag_pad_idx=-1)

    # Load the model
    model = BertForSequenceTagging.from_pretrained(tagger_model_dir)
    model.to(params.device)

    return model, data_loader, args.dataset, params


def BertNerResponse(model, queryString, bank_list):
    model, data_loader, dataset, params = model
    # if dataset in ['msra']:
    queryString = [i for i in queryString]
    # elif dataset in ['conll']:
    #     queryString = nltk.word_tokenize(queryString)

    with open('./ner_ours_test/data/' + dataset + '/interactive/sentences.txt', 'w', encoding='gbk') as f:
        f.write(' '.join(queryString))

    inter_data = data_loader.load_data('interactive')
    inter_data_iterator = data_loader.data_iterator(inter_data, shuffle=False)
    result = interAct(model, inter_data_iterator, params)
    res = {}  # 抽取出来的实体列表
    orgs = []  # 抽取出来的ORG列表
    for item in result:
        # if dataset in ['msra']:
        # res.append({''.join(queryString[item[1]:item[2]+1]):item[0]})
        res[''.join(queryString[item[1]:item[2]+1])] = item[0]
        if item[0] == 'ORG':
            cur = ''.join(queryString[item[1]:item[2]+1])
            cur = cur.strip()
            # 下面可以输出实体名称字典
            if cur not in orgs:
                orgs.append(cur)
        # elif dataset in ['conll']:
        #     res.append((' '.join(queryString[item[1]:item[2]+1]), item[0]))
    # for test
    # print(res)
    return decision(orgs, bank_list), res


def decision(cur_list, refer_list):
    for i, k in enumerate(cur_list):
        for j, refer in enumerate(refer_list):
            if k in refer or refer in k:
                return len(k) > 1


def relative(model, txt, bank_list):
    return BertNerResponse(model, txt, bank_list)


def main():
    model = bert_ner_init()
    while True:
        query = input('Input:')
        if query == 'exit':
            break
        t = time.time()
        print(BertNerResponse(model, query))
        print('time:  ', time.time()-t)


if __name__ == '__main__':
    main()
