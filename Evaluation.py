import torch


class Result_Eval(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mask = torch.zeros(1, dtype=torch.long)
        self.y_hat = torch.zeros(1, dtype=torch.long)
        self.y = torch.zeros(1, dtype=torch.long)
        self.class_dict = {1: "OBJ", 2: "SUB", 3: "ATTR", 4: "SENTI", 5: "KW"}
        self.truth = {"OBJ": 0, "SUB": 0, "ATTR": 0, "SENTI": 0, "KW": 0}
        self.prediction = {"OBJ": 0, "SUB": 0, "ATTR": 0, "SENTI": 0, "KW": 0}
        self.correct = {"OBJ": 0, "SUB": 0, "ATTR": 0, "SENTI": 0, "KW": 0}

    def add(self, output, target, attention_mask):
        self.y = torch.cat((self.y, target.to("cpu")))
        self.y_hat = torch.cat((self.y_hat, output.to("cpu")))
        attention_mask = attention_mask.view(-1)
        self.mask = torch.cat((self.mask, attention_mask.to("cpu")))

    def get_dict(self, temp_y):
        res_dict, index, n = [{}, {}, {}, {}, {}, {}], 0, temp_y.size(0)
        while index < n:
            if temp_y[index] == 0 or temp_y[index] >= 6:
                index += 1
                continue
            s_index = index
            while index < n and temp_y[index] == temp_y[s_index]:
                index += 1
            res_dict[temp_y[s_index]][s_index] = index - s_index
        return res_dict

    def eval_model(self):
        index = (self.mask == 1)
        temp_y, temp_y_hat = self.y[index], self.y_hat[index]

        y_dict = self.get_dict(temp_y)
        for c in range(1, self.num_classes):
            self.truth[self.class_dict[c]] = len(y_dict[c])

        y_hat_dict = self.get_dict(temp_y_hat)
        for c in range(1, self.num_classes):
            self.prediction[self.class_dict[c]] = len(y_hat_dict[c])

        for c in range(1, self.num_classes):
            for key, value in y_hat_dict[c].items():
                if key in y_dict[c] and value == y_dict[c][key]:
                    self.correct[self.class_dict[c]] += 1
        F_measure = 0
        for value in self.class_dict.values():
            if self.prediction[value] == 0 or self.truth[value] == 0:
                continue

            Precision = self.correct[value] / self.prediction[value]
            Recall = self.correct[value] / self.truth[value]
            if Precision == 0 and Recall == 0:
                continue
            F = 2 * Precision * Recall / (Precision + Recall)
            F_measure += F

            print("class-type {} precision value is {:.2f}".format(value, Precision))
            print("class-type {} Recall value is: {:.2f}".format(value, Recall))
            print("class-type {} F1_Measure value is: {:.2f}".format(value, F))
        print("Final F_Measure is {:.2f}".format(F_measure))