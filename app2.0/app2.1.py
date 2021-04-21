# -*- coding:utf-8 -*-
from flask import Flask, request
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AdamW, AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn import functional as F
import torch.nn as nn
import time
import json
from interactive import relative, bert_ner_init
import re
from transformers.convert_graph_to_onnx import convert
from torch.autograd import Variable
import onnx


class DataToDataset(Dataset):
    def __init__(self, encoding, labels):
        self.encoding = encoding
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.encoding['input_ids'][index], self.encoding['attention_mask'][index], self.labels[index]


def whether_effective():
    '''
    加载第二阶段模型
    '''
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_PATH = "./effective_finbert-64-32-32-256/"
    model = BertForSequenceClassification.from_pretrained(
        OUT_PATH, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained('./finbert-sentiment/')
    model.to(device)
    model.eval()

    return model, tokenizer


# def load_relative_effective():
#     '''
#     加载第一阶段模型
#     '''
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     OUT_PATH = "./relative_effective_finbert-sentiment-64-32-32-256/"
#     model = BertForSequenceClassification.from_pretrained(
#         OUT_PATH, num_labels=2)
#     tokenizer = BertTokenizer.from_pretrained('./finbert-sentiment/')
#     model.to(device)
#     model.eval()

#     return model, tokenizer


def sentiment_analysis():
    '''
    加载第三阶段模型
    '''
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_PATH = "./sentiment_analysis_finbert-more-balance64-32-32-256"
    model = AutoModelForSequenceClassification.from_pretrained(
        OUT_PATH, num_labels=3)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained('./finbert-sentiment')
    model.eval()

    return model, tokenizer


def process_data(test_sentence, tokenizer):
    '''
    预处理数据
    '''
    test_targets = np.array([0]*len(test_sentence))
    test_sentence_token = tokenizer(
        test_sentence,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    test_targets_token = torch.tensor(test_targets)
    test_dataset = DataToDataset(test_sentence_token, test_targets_token)
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=64, shuffle=False, num_workers=0)

    return test_loader


def test_effective(model, test_loader):
    '''
    预测是否有效，返回第一个是有效的概率，第二个是无效的概率
    '''
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for k, test_data in enumerate(test_loader):
        with torch.no_grad():
            test_input_ids, test_attention_mask, test_labels = [
                elem.to(device) for elem in test_data]
            test_outputs = model(test_input_ids, test_attention_mask)
            # test_out = test_outputs.logits.detach().cpu().numpy()
            softmax_out = F.softmax(test_outputs.logits, dim=1).detach().cpu().numpy()[
                0].tolist()
    return softmax_out[0], softmax_out[1]


def relative_effective(model, test_loader):
    '''
    预测是否有效，使用content判断是否有效文本
    '''
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(test_loader)
    for k, test_data in enumerate(test_loader):
        with torch.no_grad():
            test_input_ids, test_attention_mask, test_labels = [
                elem.to(device) for elem in test_data]
            test_outputs = model(test_input_ids, test_attention_mask)
            # test_out = test_outputs.logits.detach().cpu().numpy()
            softmax_out = F.softmax(test_outputs.logits, dim=1).detach().cpu().numpy()[
                0].tolist()
    # if softmax_out[0] < softmax_out[1]:
    #     return False
    # else:
    #     return True
    return softmax_out[0], softmax_out[1]


def test_sentiment(model, test_loader):
    '''
    预测情感，输出不同情感概率，依次：正负中
    '''
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t3 = time.time()
    for k, test_data in enumerate(test_loader):
        with torch.no_grad():
            test_input_ids, test_attention_mask, test_labels = [
                elem.to(device) for elem in test_data]
            test_outputs = model(test_input_ids, test_attention_mask)
            # test_out = test_outputs.logits.detach().cpu().numpy()
            softmax_out = F.softmax(
                test_outputs.logits, dim=1).detach().cpu().numpy()[0].tolist()
            print('sentiment time:  ', time.time()-t3)
            # return softmax_out[0], softmax_out[1], softmax_out[2]
            return {'positive': softmax_out[0], 'negative': softmax_out[1], 'neutral': softmax_out[2]}


def pre_process_data(s):
    # 去除数字
    # s = re.sub('[0-9]+', "", s)
    # 去除不可见字符
    s = re.sub(
        '[\001\002\003\004\005\006\007\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a]+', '', s)
    # 去除特殊字符
    s = re.sub(
        u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", s)
    return s


def predict(s):
    '''
    预测，返回不同结果预测比例
    '''
    t1 = time.time()
    res = relative(model_ner, s, bank_list)
    print('NER time: ', time.time()-t1)
    if res[0]:
        title_content = [s]
        t2 = time.time()
        test_loader_effe = process_data(title_content, tokenizer_effe)
        effect_prob, invalid_prob = test_effective(
            model_effe, test_loader_effe)
        print('effe time: ', time.time()-t2)
        if effect_prob < invalid_prob:
            return {'invalid news': invalid_prob}, res[1]
        else:
            test_loader_sent = process_data(title_content, tokenizer_sent)
            return test_sentiment(model_sent, test_loader_sent), res[1]
    else:
        return {'invalid news': 1}, res[1]


# app = Flask(__name__)


# def request_parse(req_data):
#     if req_data.method == 'POST':
#         data = req_data.json
#     elif req_data.method == 'GET':
#         data = req_data.args
#     return data


# @app.route('/predict', methods=['GET', 'POST'])
# def prodict():
#     data = request_parse(request)
#     title = data.get("title")
#     content = data.get("content")
#     result = predict(pre_process_data(title+content))
#     res = {"classify": result[0], "ner": result[1]}
#     return json.dumps(res)


# if __name__ == '__main__':
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     device = torch.device("cpu")
#     with open("./ner_ours_test/cache.json", encoding="utf-8") as f1:
#         json_file1 = json.load(f1)
#         bank_list = []
#         for d in json_file1:
#             if json_file1[d] not in bank_list:
#                 bank_list.append(json_file1[d])
#                 bank_list.append(d)
#     bank_list = set(bank_list)
#     model_effe, tokenizer_effe = whether_effective()
#     model_ner = bert_ner_init()
#     # model_rela_effe, tokenizer_rela_effe = load_relative_effective()
#     model_sent, tokenizer_sent = sentiment_analysis()
#     app.run(host='0.0.0.0', port=5000)


if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    input_name = ['input']
    output_name = ['output']
    input = Variable(torch.randn(64, 256, 768))
    
    OUT_PATH = "./effective_finbert-64-32-32-256/"
    model = BertForSequenceClassification.from_pretrained(
        OUT_PATH, num_labels=2)

    torch.onnx.export(model, input, './onnx/effective.onnx', input_names=input_name, output_names=output_name, verbose=True)
    # 检查一下生成的onnx
    test = onnx.load('./onnx/effective.onnx')
    onnx.checker.check_model(test)
    print("==> Passed")


    model_effe, tokenizer_effe = whether_effective()
    model_sent, tokenizer_sent = sentiment_analysis()
    model_ner = bert_ner_init()

    with open("./ner_ours_test/cache.json", encoding="utf-8") as f1:
        json_file1 = json.load(f1)
        bank_list = []
        for d in json_file1:
            if json_file1[d] not in bank_list:
                bank_list.append(json_file1[d])
                bank_list.append(d)
    bank_list = set(bank_list)
    # while True:
    title = '兴业银行落地全国首笔林票质押贷款'
    content = '4月2日，兴业银行三明分行向福建省三明市沙县林农杨孙忠发放149万元林票质押贷款，用于购买苗木，扩大生产，成为全国落地的首笔林票质押贷款。　　　　杨孙忠是沙县白溪口村村民，通过承包林场从事种植、培育林木等产业，由于林木从种植到采伐需要二十年左右的生长期，在此期间需要对林木进行精细养护，由于林木生长期无法采伐出售，导致杨孙忠资金紧张。　　　　“林改前，森林资源难以变现，我们林农融资难。林改后，林票能质押，兴业银行三明分行给予林农信贷支持，手上的资源盘活了，资金宽裕了，解决了我们林农的一件大事。”杨孙忠开心地说。　　　　三明市是是全国“林改”先行区，也是福建省绿色金融改革试验区，林业资源丰富，全市森林覆盖率78.7 %，森林蓄积量1.65亿立方米，林业总产值位居全国前列，但面对生态保护投入大、生态资源转化较滞后等问题，如何撬动山林“沉睡”资本，发挥林改策源地优势？　　　　2019年，三明市在全国率先开展以“合作经营、量化权益、自由流转、保底分红”为主要内容的林票制度改革试点，2020年12月全国林业改革发展综合试点市授牌仪式在三明沙县举行，标志着三明市正式成为全国首个林业改革发展综合试点市。　　　　据悉，“林票制”是指国有林业企事业单位与村集体经济组织及成员共同出资造林或合作经营现有林地，由合作双方按投资份额制发的股权（股金）凭证，具有交易、质押、继承、兑现等功能。林票质押，不仅打破森林资源流通性差的壁垒，完善林票资本权能，实现资源变资产的转换，利用金融资源配置引导实体经济绿色发展。　　　　作为中国绿色金融先行者，兴业银行发挥绿色金融领先优势和专业能力，积极协助三明市通过林票制改革，创设林票质押贷款，盘活辖区内丰富的林业资源，打通“绿水青山”向“金山银山”转化通道，让当地林农和企业从林改中真正受益，推动林业实现高质量发展。　　　　去年，兴业银行与三明市政府签订了战略合作协议，“融资+融智”支持三明绿色金融改革试验区建设，并将三明作为该行绿色金融改革创新试验田，积极推动绿色金融产品、服务、模式创新优先在三明落地，先后落地林权抵押贷、林权支贷宝等业务，大大推动了林权流转，并围绕当地特色农业、工业转型升级和绿色制造、文旅康养等10个领域进一步深化项目合作，助力三明打造“两山”理论低碳发展样板区。截至2021年3月末，兴业银行在三明投放绿色金融融资余额22.75亿元。'
    # title = input('title:')
    # content = input('content:')
    s = title+content
    s = pre_process_data(s)
    t = time.time()
    res = predict(s)
    print('time:  ', time.time()-t)
    print(res[1], res[0])
    # print('time:  ', time.time()-t)
