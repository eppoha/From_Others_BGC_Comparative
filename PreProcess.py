import os
import re
import numpy as np
from stanfordcorenlp import StanfordCoreNLP


def process_label(path, type, inverse_class_dict):
    label_col, last_doc = [], ""

    with open(path, 'r', encoding='gb18030', errors='ignore') as f:
        for line in f.readlines():
            word_list = list(filter(None, line.split()))

            # type == 0 denote train label
            if type == 0:
                doc_num = word_list[0]
                tag_col = word_list[1:]

            # type == 1 denote test label
            else:
                doc_num = word_list[1]
                tag_col = word_list[-5:]

            sentence_tag = {}
            for index, tag_word in enumerate(tag_col):
                temp_list = []
                if tag_word == "NULL":
                    continue

                if tag_word.find("|") != -1:
                    temp_list = tag_word.split('|')

                elif tag_word.find("...") != -1:
                    temp_list = tag_word.split('...')

                elif tag_word.find("、") != -1:
                    temp_list = tag_word.split('、')

                else:
                    temp_list = [tag_word]

                for token in temp_list:
                    sentence_tag[token] = inverse_class_dict[index + 1]

            if doc_num == last_doc:
                for key, value in sentence_tag.items():
                    if key not in label_col[-1]:
                        label_col[-1][key] = value
            else:
                label_col.append(sentence_tag)

            last_doc = doc_num
    return label_col


def process_data(path, tag_col, class_dict):
    input_data, target = [], []
    pattern = r'<DOC[0-9]+>\t([\s\S]*?)</DOC[0-9]+>'
    with open(path, "r", encoding='gb18030', errors='ignore') as f:
        file_data = f.read()
        data = re.findall(pattern, file_data)

        # error label col or file error
        if len(data) != len(tag_col):
            print("Please check your label collection !")

        for index, doc in enumerate(data):
            label = np.zeros(len(doc), dtype=int)

            for key, value in tag_col[index].items():
                s_index = doc.find(key)
                e_index = s_index + len(key)

                label[s_index: e_index] = class_dict[value]

            input_data.append(doc)
            target.append(label.tolist())

    return input_data, target


def char_feature(input_data, target):
    return input_data, target


def token_feature(input_data, target):
    stanford_nlp = StanfordCoreNLP('C:\\stanford-corenlp-full-2018-10-05', lang='zh')

    final_data, final_target = [], []
    for i, doc in enumerate(input_data):
        token_list, index = stanford_nlp.word_tokenize(doc), 0
        temp_target = [0] * len(token_list)

        for j in range(len(token_list)):
            class_count = np.zeros(6)
            for k in range(len(token_list[j])):
                class_count[target[i][index]] += 1
                index += 1
            temp_target[j] = np.argmax(class_count).item()

        final_data.append(token_list)
        final_target.append(temp_target)
    return final_data, final_target


def store_process_data(data, target, path):
    write_str = ""
    for i in range(len(data)):
        for j in range(len(data[i])):
            write_str += data[i][j] + ' ' + str(target[i][j]) + '\n'
        write_str += '\n'
    with open(path, 'w', encoding='utf8', errors='ignore') as f:
        f.write(write_str)


def pre_process_main():
    train_data_path = "./data/train_data.txt"
    train_label_path = "./data/train_label.txt"
    test_data_path = "./data/test_data.txt"
    test_label_path = "./data/test_label.txt"

    class_dict = {"OTHERS": 0, "OBJ": 1, "SUB": 2, "ATTR": 3, "SENTI": 4, "KW": 5}
    inverse_class_dict = {v: k for k, v in class_dict.items()}
    print(inverse_class_dict)

    # get train data label and test data label
    train_tag = process_label(train_label_path, 0, inverse_class_dict)
    test_tag = process_label(test_label_path, 1, inverse_class_dict)

    # get train data and label
    train_data, train_target = process_data(train_data_path, train_tag, class_dict)
    test_data, test_target = process_data(test_data_path, test_tag, class_dict)

    # # get token feature
    # train_final_data, train_final_target = token_feature(train_data, train_target)
    # test_final_data, test_final_target = token_feature(test_data, test_target)

    # get char feature
    train_final_data, train_final_target = char_feature(train_data, train_target)
    test_final_data, test_final_target = char_feature(test_data, test_target)

    # store pre-process data and target
    store_train_path = "./result/char_train.data"
    store_test_path = "./result/char_test.data"

    store_process_data(train_final_data, train_final_target, store_train_path)
    store_process_data(test_final_data, test_final_target, store_test_path)