class Config(object):
    def __init__(self, args):
        self.input_size = args.input
        self.hidden_size = args.hidden

        self.num_layers = args.layer
        self.num_classes = 6

        self.epochs = args.epoch
        self.batch_size = args.batch
        self.device = args.device

        self.train_data_path = args.train
        self.test_data_path = args.test
        self.bert_model_path = args.bert
        self.hidden_dropout_prob = 0.05