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
    加载第一阶段模型
    '''
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_PATH = "./effective_finbert-64-32-32-256/"
    model = BertForSequenceClassification.from_pretrained(
        OUT_PATH, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained('./finbert-sentiment/')
    model.to(device)
    model.eval()

    return model, tokenizer


def load_relative_effective():
    '''
    加载第二阶段模型
    '''
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_PATH = "./relative_effective_finbert-sentiment-64-32-32-256/"
    model = BertForSequenceClassification.from_pretrained(
        OUT_PATH, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained('./finbert-sentiment/')
    model.to(device)
    model.eval()

    return model, tokenizer


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


def predict(title, content):
    '''
    预测，返回不同结果预测比例
    '''
    s = title+content
    s = pre_process_data(s)
    t1 = time.time()
    res = relative(model_ner, s, bank_list)
    print('NER time: ', time.time()-t1)
    if res[0]:
        title_content = [s]
        # title = [pre_process_data(title)]
        # content = [pre_process_data(content)]
        t2 = time.time()
        test_loader_effe = process_data(title_content, tokenizer_effe)
        effect_prob, invalid_prob = test_effective(model_effe, test_loader_effe)
        print('effe time: ', time.time()-t2)
        if effect_prob < invalid_prob:
            return {'invalid news': invalid_prob}, res[1]
        else:
            # test_loader_rela_effe = process_data([s], tokenizer_rela_effe)
            # rela_effect_prob, rela_invalid_prob = relative_effective(model_rela_effe, test_loader_rela_effe)
            # if rela_effect_prob < rela_invalid_prob:
            #     return {'relative but invalid news': rela_invalid_prob}
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


# @app.route('/predict', methods=['GET','POST'])
# def prodict():
#     data = request_parse(request)
#     title = data.get("title")
#     content = data.get("content")
#     result = predict(title, content)
#     res = { "classify":result[0],"ner":result[1]}
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
    model_effe, tokenizer_effe = whether_effective()
    model_rela_effe, tokenizer_rela_effe = load_relative_effective()
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
    title = '反洗钱持续加码 央行年内累计罚款超亿元'
    content = '本报记者郝亚娟张荣旺上海报道2021年，反洗钱监管持续升级。4月份，央行各地分支机构公布罚单，因存在反洗钱违法违规行为，多家银行、券商、支付公司“赫然在列”。据《中国经营报》记者不完全统计，截至4月15日，因反洗钱不力被处罚的机构累计罚款金额超过1亿元。自FATF（反洗钱金融行动特别工作组）第四轮互评以来，央行明显加快了对金融机构和特定非金融机构的反洗钱反恐融资监管力度。然而，在数字时代，由于涉及数据范围广、分布散、历史数据多、业务场景复杂，金融机构在开展反洗钱治理中仍面临严峻挑战。“严”字当头近年来，反洗钱成为重点监管领域，亦是不少金融机构、支付机构受罚的重灾区。按照《反洗钱法》，在我国应履行反洗钱义务的金融机构，包括政策性银行、商业银行、信用合作社、邮政储汇机构、信托公司、证券公司、期货经纪公司、保险公司及央行确定并公布的从事金融业务的其他机构等。央行方面指出，2020年央行对614家金融机构、支付机构等反洗钱义务机构开展了专项和综合执法检查，依法完成对537家义务机构的行政处罚，处罚金额5.26亿元，处罚违规个人1000人，处罚金额2468万元。普华永道相关分析报告指出，2020年度的733笔罚单中，195笔处罚是对多项违规行为进行综合处罚，538笔处罚是对单项违规行为进行处罚。“未按规定履行客户身份识别义务”为首要处罚原因，共涉及547笔罚单，合计涉及处罚金额约4.3亿元；其次是“未按规定报送大额交易报告或可疑交易报告”，共涉及225笔罚单，合计涉及处罚金额约3.2亿元。相比2020年的处罚情况，今年的反洗钱力度继续加码。2月25日，央行2021年反洗钱工作电视会议在北京召开，会议提出积极推进《反洗钱法》修订工作进程，不断完善反洗钱制度体系，大力加强反洗钱协调机制建设，提升各部门反洗钱工作合力。值得一提的是，中信银行在今年2月被处罚款2890万元，14名相关责任人也一并被罚，成为今年以来因反洗钱受罚金额最大的银行。从处罚事由来看，包括未按规定履行客户身份识别义务；未按规定保存客户身份资料和交易记录；未按规定报送大额交易报告或可疑交易报告；与身份不明的客户进行交易。这四项违法事由均踩了《反洗钱法》的红线。4月以来，央行西安分行、长春中心支行等分支机构亦陆续公布反洗钱罚单。其中，长安银行因违反支付结算、反洗钱、货币金银、国库、征信管理规定，被给予警告并处420.8万元罚款。甘肃银行、渤海银行、吉林亿联银行、3家农商行及相关负责人分别被处以罚款和警告。此外，随着移动互联网的迅速发展，反洗钱监管工作重点也逐步向非银行金融机构转移，小贷公司、支付机构也是重点关注对象。今年2月，因涉及踩反洗钱红线等十宗违法行为，第三方支付公司重庆市钱宝科技服务有限公司连同相关责任人被监管部门处以罚款金额超900万元。3月31日，中国人民银行、银保监会、证监会发布《金融机构客户尽职调查和客户身份资料及交易记录保存管理办法（修订草案征求意见稿）》（以下简称“《办法》”），对金融机构及支付机构的反洗钱责任有了更加详细的要求。《办法》扩大了适用范围，增加非银行支付机构、网络小额贷款公司、银行理财子公司等从事金融业务的机构；要求银行机构履行客户尽职调查义务时，按照法律、行政法规或部门规章的规定需核实相关自然人的居民身份证或者非居民身份证件的，应当通过中国人民银行建立的联网核查公民身份信息系统进行核查。一位反洗钱专家指出，洗钱危害金融体系的安全，在洗钱活动中，资金的走向完全脱离正常交易的收支特点，毫无规律可循，银行不能按照资金运作规律来调度头寸，严重时会引发支付危机，并会扰乱资源配置和扭曲市场。方达律师事务所资深律师汪灵罡告诉记者，自FATF第四轮互评以来，央行明显加快了对金融机构和特定非金融机构的反洗钱反恐融资监管力度。一方面，以双罚制的严厉处罚震慑金融机构；另一方面，加快了修法立规的节奏，央行试图努力把中国金融业反洗钱反恐融资水平在较短时间内提高到FATF期望的高度。技术难关“洗钱方式在不断迭代，传统的作案手法主要有利用公司和控制他人账户洗钱、利用境外账户洗钱、将非法资金进行投资理财等。”前述受访反洗钱专家指出。值得注意的是，在反洗钱治理中，银行类金融机构仍是受罚主体。“由于商业银行服务的客户群体大，各机构、各地区发展水平和风险意识程度不一，在短期内实现反洗钱反恐融资风险管理水平飞跃和质变的难度不低。无论是个人业务、公司业务，还是金融市场业务，理论上银行的每一项业务都有可能成为洗钱的渠道。”汪灵罡指出。以私人银行业务为例，因其具有客户高端性、信息私密性、业务综合性等特征，使其成为反洗钱风险的“多发之地”，也给银行的私人银行客户身份识别工作带来难度。一般来说，银行反洗钱系统会自动筛选出大额交易和异常交易。针对异常交易，银行须结合客户基本信息和交易背景等要素，进行人工分析和甄别。只要发现具备合理理由怀疑客户本身、客户的资金和其他资产、客户的交易或正准备进行的交易，与洗钱、恐怖融资等犯罪活动有关，不论资金金额或资产价值大小，银行都应当提交可疑交易报告。某城商行人士告诉记者，按照相关要求，无论是存续业务还是新业务，该行都要做大额资金管控，账户流水和企业规模匹配，要求企业开户限额以实缴资金为准，后续提额需要具体的业务合同和证明材料。“如果账户资金基本为流入，且为非贸易性流入、流入来源广泛、与其经营无相关性，而流出较少或非正常贸易流出，银行就会加强监测。”他举例道。“尽管商业银行已经投入巨量资金和人力资源去积极提高反洗钱反恐融资的风险管理水平，但仍存在一些困难，如基层网点的业务考核压力大、可疑交易监控与分析需要相关工作人员经年累月积累经验、人才培养严重不足、专业软件服务商水平不高等。”汪灵罡坦言。尤其在互联网金融形势下，商业银行反洗钱工作更为严峻。“随着业务流程线上化，银行传统的准入方法很难识别。以申请信用卡为例，在批量申请中，只要满足银行某一维度的要求，洗钱团伙很可能借此蒙混过关。对于银行而言，如果未能及时发现会有两个后果，一是反洗钱不力，违反监管要求；二是资金在短时间内流转出且很难追回，最终形成坏账。”某受访金融科技公司人士表示。记者采访了解到，各家银行加大反洗钱工作的资源投入，并在知识图谱、风控模型上与金融科技公司进行合作。该金融科技公司人士告诉记者：“目前大型银行基本有自己的反洗钱团队，不过也会与金融科技公司合作；一些城商行主要是找外包公司合作反洗钱。从我们接触的来看，识别团伙作案是各家金融机构反洗钱治理的迫切需求。”“利用空壳公司虚假开户、利用民工身份证骗取公积金贷款、互联网平台挪用公众资金等洗钱行为多发。目前来看，金融机构对自身面临的洗钱风险的认识不全面。以前出于合规性要求，根据政策制定自己的管理框架。但随着技术发展，金融机构面临的外部风险是不断变化的，这就要求机构加强洗钱风险自评估指引，从客户、区域、产品、渠道等不同维度判断自己固有的反洗钱风险，并同步采取措施改进。”一位资深银行人士告诉记者。中国人民银行反洗钱局局长巢克俭此前在公开场合表示，要大力提高反洗钱和反恐怖融资领域金融科技水平，进一步加强与科技行业、金融机构等合作，推动金融科技和监管科技在反洗钱和反恐怖融资领域的应用和发展。普华永道建议，反洗钱义务机构需加强以下四个方面：一是重视反洗钱数据治理工作，从数据出发，对数据完整性、合理性进行校验，以发现客户身份识别不到位、交易行为可疑等情形；二是完善客户尽职调查机制，以风险为本，建立清晰的客户接纳政策，在持续尽职调查的要求中融入客户全生命周期管理，采取定期审核与事件触发相结合的方式，确定客户持续尽职调查开展的频率和方式，并结合持续尽调情况调整客户风险等级；三是搭建定制化机构洗钱风险评估体系；四是升级业务洗钱风险评估工作，针对识别出的固有风险，开展相对应的控制措施有效性评估，确保真正发现风险管理漏洞，并及时完善控制措施，真正做到从“规则为本”向“风险为本”的转变。'
    # title = input('title:')
    # content = input('content:')
    t = time.time()
    res = predict(title, content)
    print('time:  ', time.time()-t)
    print(res[1], res[0])
    # print('time:  ', time.time()-t)
