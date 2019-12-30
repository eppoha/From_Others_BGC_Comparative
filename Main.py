import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import argparse
import Model
import MyData
import DataProcess
# import PreProcess
import Config
import Evaluation
import matplotlib.pyplot as plt
import matplotlib
import Bert_GRU_CRF_Model

from pytorch_pretrained_bert import BertAdam


# padding data and return input_ids, token_type_ids, attention_mask
def padding_data(data, target, maxLen):
    input_ids = [line + [0] * (maxLen - len(line)) for line in data]
    token_type_ids = [[0] * maxLen for _ in data]
    attention_mask = [[1] * len(line) + [0] * (maxLen - len(line)) for line in data]
    target_col = [line + [-1] * (maxLen - len(line)) for line in target]
    return input_ids, token_type_ids, attention_mask, target_col


def TrainModel(model, optimizer, train_loader, maxLen, device, epoch):
    epoch_loss, t = 0, 0
    for index, (data, target) in enumerate(train_loader):
        # through data to get input_ids, token_type_ids, attention_mask
        if index >= 10 and index % 10 == 0:
            print("index: ", index)
        input_ids, token_type_ids, attention_mask, target_col = padding_data(data, target, maxLen)

        input_ids = torch.tensor(input_ids).long().to(device)
        token_type_ids = torch.tensor(token_type_ids).long().to(device)
        attention_mask = torch.tensor(attention_mask).long().to(device)
        target_col = torch.tensor(target_col).to(device)

        loss = model(input_ids, token_type_ids, attention_mask, target_col)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t = index

    print("Loss: {}".format(epoch_loss / t))


def TestModel(model, test_loader, maxLen, device, epoch):
    total, correct = 0, 0
    res_eval = Evaluation.Result_Eval(6)
    with torch.no_grad():
        for index, (data, target) in enumerate(test_loader):
            # through data to get input_ids, token_type_ids, attention_mask
            input_ids, token_type_ids, attention_mask, target_col = padding_data(data, target, maxLen)

            input_ids = torch.tensor(input_ids).long().to(device)
            token_type_ids = torch.tensor(token_type_ids).long().to(device)
            attention_mask = torch.tensor(attention_mask).long().to(device)
            target_col = torch.tensor(target_col).to(device)

            target_col = target_col[attention_mask == 1]

            output = model(input_ids, token_type_ids, attention_mask)

            total += torch.sum(attention_mask == 1).item()
            correct += torch.sum(target_col == output).item()
            attention_mask = attention_mask[attention_mask == 1]

            res_eval.add(output, target_col, attention_mask)
        res_eval.eval_model()
        print("ACC: {:.2f}%, correct/total: {}/{}".format(correct / total * 100, correct, total))


def TerminalParser():
    # define train data path and test data path
    train_data_path = "./result/char_train.data"
    test_data_path = "./result/char_test.data"
    bert_model_path = "/home/zhliu/RTX/modeling_bert/bert-base-chinese.tar.gz"
    # bert_model_path = "C:\\Users\\curry\\Desktop\\modeling_bert\\bert-base-chinese.tar.gz"

    # define parse parameters
    parser = argparse.ArgumentParser()
    parser.description = 'choose train data and test data file path'
    parser.add_argument('--train', help='train data file path', default=train_data_path)
    parser.add_argument('--test', help='test data file path', default=test_data_path)
    parser.add_argument('--bert', help='bert model file path', default=bert_model_path)
    parser.add_argument('--batch', help='input data batch size', default=5)
    parser.add_argument('--input', help='input data size', default=768)
    parser.add_argument('--hidden', help='gru hidden size', default=100)
    parser.add_argument('--layer', help='the number of gru layer', default=3)
    parser.add_argument('--epoch', help='the number of run times', default=100)
    parser.add_argument('--device', help='run program in device type', default='cuda')
    # parser.add_argument('--device', help='run program in device type', default='cpu')
    args = parser.parse_args()

    config_obj = Config.Config(args)
    return config_obj


def main():
    # get some configure and hyper-parameters
    args = TerminalParser()

    # get standard data file
    if not os.path.exists(args.train_data_path) or not os.path.exists(args.test_data_path):
        PreProcess.pre_process_main()

    train_path = "./result/char_train.data"
    test_path = "./result/char_test.data"

    # get vocab and max sequence length
    vocab, maxLen = DataProcess.get_vocab(train_path, test_path)
    maxLen += 10

    # get train data and test data BERT
    train_data, train_target = DataProcess.get_bert_data(train_path)
    test_data, test_target = DataProcess.get_bert_data(test_path)

    # define train data and test data
    train_loader = MyData.get_loader(train_data, train_target, args.batch_size)
    test_loader = MyData.get_loader(test_data, test_target, args.batch_size)

    # define model and optimizer
    model = Bert_GRU_CRF_Model.BGCM(args).to(args.device)
    optimizer = optim.Adam([{'params': model.fc2.parameters(), 'lr': 0.001},
                            {'params': model.fc1.parameters(), 'lr': 0.001},
                            {'params': model.bert.parameters(), 'lr': 2e-5},
                            {'params': model.gru.parameters(), 'lr': 0.001},
                            {'params': model.crf.parameters(), 'lr': 0.001}], weight_decay=0.01)

    # train model and test model
    for epoch in range(args.epochs):
        TrainModel(model, optimizer, train_loader, maxLen, args.device, epoch)
        TestModel(model, test_loader, maxLen, args.device, epoch)


if __name__ == "__main__":
    main()
