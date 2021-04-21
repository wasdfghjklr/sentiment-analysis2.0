from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel, AdamW, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForPreTraining
from torch.nn import functional as F
import torch.nn as nn
import os
import time
import argparse
import logging

# parser for hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--log', type=str, default='debug', help='{info, debug}')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of epochs to train.')
args = parser.parse_args()

# logger
logging.basicConfig(format='%(message)s',
                    level=getattr(logging, args.log.upper()))


def drop_nan(df):
    # 排空
    df.dropna(subset=['content'], inplace=True)
    df.dropna(subset=['relativity'], inplace=True)
    return df


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)


def get_data(df):
    sentences = list(df['content'])
    targets = df['relativity'].values
    return sentences, targets


def tokenizer_data(sentences, targets):
    sentences_tokened = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    targets = torch.tensor(targets)

    return sentences_tokened, targets


class DataToDataset(Dataset):
    def __init__(self, encoding, labels):
        self.encoding = encoding
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.encoding['input_ids'][index], self.encoding['attention_mask'][index], self.labels[index]

# OUT_PATH = "./results_model/finebert-64-32-32-256/"
# model_name = "../sentiment_analysis/finbert-sentiment"
# model = AutoModelForSequenceClassification.from_pretrained(
#     OUT_PATH, num_labels=2)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# OUT_PATH = "./results_model/bert-base-chinese-64-32-32-256"
# model_name = "bert-base-chinese"
# model = BertForSequenceClassification.from_pretrained(
#     OUT_PATH, num_labels=2)
# tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)

# OUT_PATH = "./results_model/electra-180g-small-64-32-32-256"
# MODEL_NAME = "hfl/chinese-electra-180g-small-discriminator"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForSequenceClassification.from_pretrained(OUT_PATH, num_labels=2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


#----------load data----------#
test_df = pd.read_excel(
    './test_data/华夏-北京-副本.xls')
test_df = test_df[['content', 'relativity']]
test_df = drop_nan(test_df)

# config: effective 0, ineffective 1
# test_df.relativity = test_df.relativity.replace(1, 0)
# test_df.relativity = test_df.relativity.replace(2, 1)

test_sentences, test_targets = get_data(test_df)

test_sentences_token, test_targets_token = tokenizer_data(
    test_sentences, test_targets)
test_dataset = DataToDataset(test_sentences_token, test_targets_token)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=32, shuffle=False, num_workers=0)
# torch.save(test_loader,
#            './test_dataloader/华夏-北京-electra-180g-small-64-32-32-256.pt')


test_acc = 0.0
test_loss = 0.0
test_f1 = 0.0
# logging.debug("testing...")
start = time.time()
# test_loader = torch.load(
#     './test_dataloader/华夏-北京-electra-180g-small-64-32-32-256.pt')
for k, test_data in enumerate(test_loader):
    model.eval()
    with torch.no_grad():
        test_input_ids, test_attention_mask, test_labels = [
            elem.to(device) for elem in test_data]
        test_outputs = model(test_input_ids, test_attention_mask)
        test_loss_cur = F.cross_entropy(test_outputs.logits, test_labels)
        test_loss += test_loss_cur.item()
        test_out = test_outputs.logits.detach().cpu().numpy()
        test_labels = test_labels.detach().cpu().numpy()
        test_acc += flat_accuracy(test_out, test_labels)
        test_preds = np.argmax(test_out, axis=1)
        test_f1 += f1_score(test_labels, test_preds, average='macro')
        
        softmax_out = F.softmax(test_outputs.logits)
        print(softmax_out)
        print('true: ', test_labels.tolist())
        print('pred: ', test_preds.tolist())

        logging.debug('test time for each instance: ')
        logging.debug((time.time()-start)/len(test_labels))
        start = time.time()
print(test_loss/(k+1), test_acc/(k+1), test_f1/(k+1))
logging.debug("test ave loss:%3f, test Acc:%3f, test f1:%3f" %
              (test_loss/(k+1), test_acc/(k+1), test_f1/(k+1)))

print('over')
