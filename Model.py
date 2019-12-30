import torch
import torch.nn as nn

from pytorch_pretrained_bert import BertModel, BertTokenizer


class BERT_Cell(nn.Module):
    def __init__(self, model_path):
        super(BERT_Cell, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.hidden_size = self.bert.config.hidden_size
        self.hidden_dropout_prob = self.bert.config.hidden_dropout_prob

    def forward(self, input_ids, token_type_ids, attention_mask):
        self.bert.eval()
        # encoded_layers shape: [layers, batch_size, tokens, hidden_size]
        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask)

        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(encoded_layers, dim=0)

        # token_embeddings size is [encode_layers, batch_size, seq_len, hidden_size]
        token_embeddings = token_embeddings.permute(1, 2, 0, 3)
        # x_embedding = torch.cat((token_embeddings[:, :, -4, :], token_embeddings[:, :, -3, :],
        #                          token_embeddings[:, :, -2, :], token_embeddings[:, :, -1, :]), dim=2)
        x_embedding = torch.cat((token_embeddings[:, :, -1, :], token_embeddings[:, :, -2, :]), dim=2)
        # x_embedding = token_embeddings[:, :, -1, :]
        return x_embedding


class BiGRU_Cell(nn.Module):
    def __init__(self, config):
        super(BiGRU_Cell, self).__init__()
        # define hyper-parameters
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.num_layer = config.num_layers
        self.batch_size = config.batch_size
        self.device = config.device

        self.gru = nn.GRU(config.input_size, config.hidden_size, config.num_layers,
                          batch_first=True, bidirectional=True)

    def forward(self, x):
        h0 = torch.zeros(self.num_layer * 2, x.size(0), self.hidden_size).to(self.device)

        # x shape is [batch_size, token_size, bert_embedding_size]
        output, _ = self.gru(x, h0)
        return output


class CRF_Cell(nn.Module):
    def __init__(self, config):
        super(CRF_Cell, self).__init__()
        # tag_size = num_classes + start tag and stop tag
        self.tag_size = config.num_classes + 2
        self.device = config.device
        self.f = nn.Sigmoid()
        self.target_to_index = {"OTHERS": 0, "OBJ": 1, "SUB": 2, "ATTR": 3,
                                "SENTI": 4, "KW": 5, "START": 6, "STOP": 7}
        self.transitions = nn.Parameter(torch.ones(self.tag_size, self.tag_size), requires_grad=True)

    # change the type
    def log_sum_exp(self, vec):
        max_score = vec[0, torch.argmax(vec, dim=1)]
        n, m = vec.size(0), vec.size(1)
        max_score_broadcast = max_score.view(-1, 1).expand(n, m)
        # print(max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast))))
        return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

    def viterbi_decode(self, word_feature, input_ids):
        index, decode_ids, each_seq_ids = 0, [], []
        forward_var = torch.full((1, self.tag_size), -1000000).to(self.device)

        # '[CLS]' id is 101 and '[SEP]' id is 102
        for feat in word_feature:
            step_best_ids = []  # store current step max value's last tag
            step_best_value = []  # store current step each tag max value

            # if feat is '[CLS]' init forward_var
            if input_ids[index] == 101:
                forward_var = torch.full((1, self.tag_size), -100000).to(self.device)
                forward_var[0][self.target_to_index['START']] = 0
                index += 1
                continue

            # if feature is "[SEP]" need get sequence tags
            if input_ids[index] == 102:
                seq_stop_var = forward_var + self.transitions[self.target_to_index['STOP']]
                current_id = torch.argmax(seq_stop_var).item()

                # find the best tag path
                path_ids = [current_id]
                for i in range(len(each_seq_ids) - 1, 0, -1):
                    step_ids = each_seq_ids[i]
                    path_ids.insert(0, step_ids[path_ids[0]])

                # 6, 7 denote start and stop
                path_ids.insert(0, 6)
                path_ids.append(7)

                # add each sequence tags to all tags
                decode_ids.extend(path_ids)
                each_seq_ids = []
                index += 1
                continue

            # else using viterbi algorithm
            for next_tag in range(self.tag_size):
                current_var = forward_var + self.transitions[next_tag]
                current_best_id = torch.argmax(current_var)

                step_best_ids.append(current_best_id)
                step_best_value.append(current_var[0][current_best_id].view(1))

            forward_var = (feat + torch.cat(step_best_value)).view(1, -1)
            each_seq_ids.append(step_best_ids)
            index += 1
        return torch.tensor(decode_ids, dtype=torch.long).to(self.device)

    def truth_path_loss(self, word_feature, input_ids, tag_col):
        # score should be loss back
        batch_score, seq_score = torch.zeros(1).to(self.device), torch.zeros(1).to(self.device)

        # using 6 and 7 instead of -1
        tag_col[input_ids == 101] = 6
        tag_col[input_ids == 102] = 7

        for index, feat in enumerate(word_feature):
            if tag_col[index] == 6:
                seq_score = torch.zeros(1).to(self.device)
                continue

            # end the sequence tag score
            if tag_col[index] == 7:
                seq_score -= self.transitions[tag_col[index], tag_col[index - 1]]
                batch_score += seq_score
                continue

            # seq_score = seq_score + self.f(feat[tag_col[index]]) + self.f(self.transitions[tag_col[index + 1], tag_col[index]])
            seq_score = seq_score + feat[tag_col[index]] + self.transitions[tag_col[index + 1], tag_col[index]]

        return batch_score

    def all_path_loss(self, word_feature, input_ids):
        forward_var = torch.zeros(self.tag_size, 1).to(self.device)
        all_path_score = torch.zeros(1).to(self.device)

        for index, feat in enumerate(word_feature):
            if input_ids[index] == 101:
                # forward_var = feat.view(self.tag_size, 1)
                forward_var = torch.zeros(self.tag_size, 1).to(self.device)

                # forward_var = torch.full((1, self.tag_size), -100000).to(self.device)
                # forward_var[0][self.target_to_index['START']] = 0
                continue

            if input_ids[index] == 102:
                forward_var = forward_var.view(1, self.tag_size)
                all_path_score += self.log_sum_exp(forward_var)
                continue

            # expand to n * n
            forward_var = forward_var.expand(self.tag_size, self.tag_size)
            # feat = self.f(feat.view(1, -1))
            feat = feat.view(1, -1)
            emission = feat.expand(self.tag_size, self.tag_size)

            # score = forward_var + emission + self.f(torch.t(self.transitions))
            score = forward_var + emission + torch.t(self.transitions)
            forward_var = self.log_sum_exp(score).view(self.tag_size, 1)

        return all_path_score

    def loss(self, word_feature, input_ids, target_col):
        truth_path_score = self.truth_path_loss(word_feature, input_ids, target_col)
        all_path_score = self.all_path_loss(word_feature, input_ids)
        return all_path_score - truth_path_score
