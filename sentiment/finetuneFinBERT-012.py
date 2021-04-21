from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModel, AdamW, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
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

# model_name = "../sentiment_analysis/finbert-sentiment"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# mymodel = AutoModelForSequenceClassification.from_pretrained(
#     model_name, num_labels=3)

model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
mymodel = BertForSequenceClassification.from_pretrained(
    model_name, num_labels=3)

# MODEL_NAME = "hfl/chinese-electra-180g-small-discriminator"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# mymodel = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3) 

# 获取gpu和cpu的设备信息
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mymodel.to(device)


#----------load data----------#
train_df = pd.read_csv(
    './dataset/train.csv', usecols=['content', 'relativity'])
val_df = pd.read_csv(
    './dataset/dev.csv', usecols=['content', 'relativity'])
test_df = pd.read_csv(
    './dataset/test.csv', usecols=['content', 'relativity'])

train_df = drop_nan(train_df)
val_df = drop_nan(val_df)
test_df = drop_nan(test_df)

print(train_df.shape, val_df.shape, test_df.shape)

# config: positive 0, negative 1, neutral 2
train_df.relativity = train_df.relativity.replace(0, 2)
val_df.relativity = val_df.relativity.replace(0, 2)
test_df.relativity = test_df.relativity.replace(0, 2)
train_df.relativity = train_df.relativity.replace(1, 0)
val_df.relativity = val_df.relativity.replace(1, 0)
test_df.relativity = test_df.relativity.replace(1, 0)
train_df.relativity = train_df.relativity.replace(-1, 1)
val_df.relativity = val_df.relativity.replace(-1, 1)
test_df.relativity = test_df.relativity.replace(-1, 1)

# -balance
train_df2 = train_df[train_df.relativity == 2].sample(n=5000, replace=False, random_state=None, axis=None)
train_df01 = train_df[train_df.relativity != 2]
train_df = pd.concat([train_df01, train_df2])


val_df2 = val_df[val_df.relativity == 2].sample(n=2000, replace=False, random_state=None, axis=None)
val_df01 = val_df[val_df.relativity != 2]
val_df = pd.concat([val_df01, val_df2])

test_df2 = test_df[test_df.relativity == 2].sample(n=2000, replace=False, random_state=None, axis=None)
test_df01 = test_df[test_df.relativity != 2]
test_df = pd.concat([test_df01, test_df2])

train_sentences, train_targets = get_data(train_df)
val_sentences, val_targets = get_data(val_df)
test_sentences, test_targets = get_data(test_df)

del train_df
del val_df
del test_df

print('Start token encodding...')

#---------token encodding----------#

train_sentences_token, train_targets_token = tokenizer_data(
    train_sentences, train_targets)
val_sentences_token, val_targets_token = tokenizer_data(
    val_sentences, val_targets)
test_sentences_token, test_targets_token = tokenizer_data(
    test_sentences, test_targets)

del train_sentences
del train_targets
del val_sentences
del val_targets
del test_sentences
del test_targets

print('Start encoding data...')

# --------encoding data---------#
# 封装数据
train_dataset = DataToDataset(train_sentences_token, train_targets_token)
val_dataset = DataToDataset(val_sentences_token, val_targets_token)
test_dataset = DataToDataset(test_sentences_token, test_targets_token)

del train_sentences_token
del train_targets_token
del val_sentences_token
del val_targets_token
del test_sentences_token
del test_targets_token

# BATCH_SIZE = 32

print('Statr data loader and save data...')

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=64, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset=val_dataset,
                        batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=32, shuffle=True, num_workers=0)

del train_dataset
del val_dataset
del test_dataset

torch.save(train_loader, './datasetloader/train_loader-bert-base-chinese-more-balance64-32-32-256.pt')
torch.save(val_loader, './datasetloader/val_loader-bert-base-chinese-more-balance64-32-32-256.pt')
torch.save(test_loader, './datasetloader/test_loader-bert-base-chinese-more-balance64-32-32-256.pt')

del train_loader
del val_loader
del test_loader

#---------train model---------#
OUT_PATH = "./three_results_model/bert-base-chinese-more-balance64-32-32-256/"
if not os.path.isdir(OUT_PATH):
    os.mkdir(OUT_PATH)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = AdamW(mymodel.parameters(), lr=1e-5)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return accuracy_score(labels_flat, pred_flat)

print('Start training! ')
epochs = args.epochs
best_acc = 0
best_loss = 1e10
for epoch in range(epochs):
    t = time.time()
    train_loss = 0.0
    train_acc = 0.0
    f1 = 0.0
    # logging.debug("training...")
    train_loader = torch.load('./datasetloader/train_loader-bert-base-chinese-more-balance64-32-32-256.pt')
    for i, data in enumerate(train_loader):
        mymodel.train()
        if epoch == 0:
            logging.debug('training batch '+str(i)+' !')
        input_ids, attention_mask, labels = [elem.to(device) for elem in data]
        optimizer.zero_grad()
        outputs = mymodel(input_ids, attention_mask)
        # softmax_out = F.softmax(outputs.logits)
        # 计算误差
        loss = F.cross_entropy(outputs.logits, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        # 计算acc
        out = outputs.logits.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        train_acc += flat_accuracy(out, labels)
        preds = np.argmax(out, axis=1)
        f1 += f1_score(labels, preds, average='macro')
    logging.debug("train %d/%d epochs Loss:%3f, Acc:%3f, f1:%3f" %
                  (epoch+1, epochs, train_loss/(i+1), train_acc/(i+1), f1/(i+1)))

    del train_loader

    #----------evaluate----------#
    # logging.debug("evaluating...")
    val_loss = 0.0
    val_acc = 0.0
    val_f1 = 0.0
    val_loader = torch.load('./datasetloader/val_loader-bert-base-chinese-more-balance64-32-32-256.pt')
    for j, batch in enumerate(val_loader):
        mymodel.eval()
        if epoch == 0:
            logging.debug('validation batch '+str(j)+' !')
        with torch.no_grad():  # 不计算梯度
            val_input_ids, val_attention_mask, val_labels = [
                elem.to(device) for elem in batch]
            # with torch.no_grad():
            pred = mymodel(val_input_ids, val_attention_mask)
            val_loss_cur = F.cross_entropy(pred.logits, val_labels)
            val_loss += val_loss_cur
            pred = pred.logits.detach().cpu().numpy()
            val_labels = val_labels.detach().cpu().numpy()
            val_preds = np.argmax(pred, axis=1)
            val_acc += flat_accuracy(pred, val_labels)
            val_f1 += f1_score(val_labels, val_preds, average='macro')
            # print(j)
    logging.debug("evaluate loss:%3f, Acc:%3f, f1:%3f" %
                  (val_loss/(j+1), val_acc/(j+1),val_f1/(j+1)))

    del val_loader

    # save model
    if best_acc < val_acc:
        best_acc = val_acc
        mymodel.save_pretrained(OUT_PATH)
    if best_loss > val_loss:
        best_loss = val_loss
        mymodel.save_pretrained(OUT_PATH)
    
    logging.debug("epoch time: ")
    logging.debug(time.time()-t)


#----------test----------#
# load model saved before
model = AutoModelForSequenceClassification.from_pretrained(
    OUT_PATH, num_labels=3)
model.to(device)


test_acc = 0.0
test_loss = 0.0
test_f1 = 0.0
# logging.debug("testing...")
start = time.time()
test_loader = torch.load('./datasetloader/test_loader-bert-base-chinese-more-balance64-32-32-256.pt')
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
        logging.debug('test time for each instance: ')
        logging.debug((time.time()-start)/len(test_labels))
        start = time.time()
print(test_loss/(k+1), test_acc/(k+1), test_f1/(k+1))
logging.debug("test ave loss:%3f, test Acc:%3f, test f1:%3f" %
              (test_loss/(k+1), test_acc/(k+1), test_f1/(k+1)))

print('over')
