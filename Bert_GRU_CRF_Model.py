import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import Main
import Evaluation
from Model import BERT_Cell, BiGRU_Cell, CRF_Cell


class BGCM(nn.Module):
    def __init__(self, config):
        super(BGCM, self).__init__()
        self.bert = BERT_Cell(config.bert_model_path)
        self.dropout = nn.Dropout(0.1)

        self.gru = BiGRU_Cell(config)
        self.fc1 = nn.Linear(config.input_size, 2 * config.hidden_size)
        self.fc2 = nn.Linear(2 * config.hidden_size, config.num_classes + 2)
        self.crf = CRF_Cell(config)

    def forward(self, input_ids, token_type_ids, attention_mask, target_col=None):
        x_word_embedding = self.bert(input_ids, token_type_ids, attention_mask)
        x_word_embedding = self.dropout(x_word_embedding)

        bert_feature = self.fc1(x_word_embedding)
        word_feature = self.gru(x_word_embedding)

        emission_prob = self.fc2(word_feature + bert_feature)

        features = emission_prob[attention_mask == 1]
        sequence_ids = input_ids[attention_mask == 1]

        if target_col is None:
            output = self.crf.viterbi_decode(features, sequence_ids)
            return output

        else:
            sequence_tags = target_col[attention_mask == 1]
            loss = self.crf.loss(features, sequence_ids, sequence_tags)
            return loss