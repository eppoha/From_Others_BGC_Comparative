import os
import numpy as np
import matplotlib.pyplot as plt

from pytorch_pretrained_bert import BertTokenizer


# get vocab and calculate the sequence max length
def get_vocab(train_path, test_path):
    maxLen, s_index = 0, 0
    vocab, index = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[UNK]': 3}, 5

    with open(train_path, "r", encoding='utf8', errors='ignore') as f:
        for i, line in enumerate(f.readlines()):
            if line == '\n':
                maxLen = max(i - s_index, maxLen)
                s_index = i + 1
                continue

            word = line.split()[0]
            if word not in vocab:
                vocab[word] = index
                index += 1

    with open(test_path, "r", encoding='utf8', errors='ignore') as f:
        for line in f.readlines():
            if line == '\n':
                continue

            word = line.split()[0]
            if word not in vocab:
                vocab[word] = index
                index += 1
    return vocab, maxLen


def get_bert_label(setence, line_target):
    tag_ids, index = [], 0
    for word in setence:
        temp_token = ""
        # special process for start flag and end flag
        if word == '[CLS]' or word == '[SEP]':
            tag_ids.append(-1)
            continue

        if word == '[UNK]':
            tag_ids.append(0)
            index += 1
            continue

        if word.find("##") != -1:
            temp_token = word.split("##")[1]
        else:
            temp_token = word

        class_count = np.zeros(6)
        for i in range(len(temp_token)):
            class_count[line_target[index]] += 1
            index += 1
        tag_ids.append(np.argmax(class_count).item())

    if len(setence) != len(tag_ids):
        print("data error!!!")

    return tag_ids


def get_bert_data(path, vocab_path='bert-base-chinese'):
    print("generate bert data representation")
    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    data_col, target_col = [], []
    with open(path, "r", encoding='utf8', errors='ignore') as f:
        sentence, line_data, line_target = "", [], []
        for index, line in enumerate(f.readlines()):
            if line == '\n':
                sentence = "[CLS] " + sentence + " [SEP]"
                line_data = tokenizer.tokenize(sentence)
                line_target = get_bert_label(line_data, line_target)
                line_data = tokenizer.convert_tokens_to_ids(line_data)

                data_col.append(line_data)
                target_col.append(line_target)
                sentence, line_data, line_target = "", [], []
                continue

            temp_list = list(filter(None, line.split(' ')))

            if temp_list[0] == u'\u3000':
                continue

            sentence += temp_list[0]
            line_target.append(int(temp_list[1].rstrip('\n')))
    return data_col, target_col


def get_gru_data(path, vocab):
    data, target = [], []
    with open(path, "r", encoding='utf8',errors='ignore') as f:
        word_ids, tag_ids = [], []
        for index, line in enumerate(f.readlines()):
            if line == '\n':
                data.append(word_ids)
                target.append(tag_ids)
                word_ids, tag_ids = [], []
                continue

            temp_list = list(filter(None, line.split(' ')))

            if temp_list[0] not in vocab:
                continue

            word_ids.append(vocab[temp_list[0]])
            tag_ids.append(int(temp_list[1].rstrip('\n')))

    return data, target


def plot_number_token(data):
    temp = [0] * 160
    label_col = [i for i in range(160)]
    for i in range(len(data)):
        temp[len(data[i])] += 1

    plt.title("The number of comparative sentences' token")
    plt.bar(label_col, temp)
    plt.xlabel("token number")
    plt.ylabel("count")
    plt.show()