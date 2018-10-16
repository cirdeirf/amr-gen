#  import warnings
#  warnings.filterwarnings('ignore')

import socket
import numpy as np
import gluonnlp as nlp
import mxnet as mx

class NNLanguageModelClient:
    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(("localhost", 32000))
        self.f = self.s.makefile('rw')
        self.lm = NNLanguageModel('awd_lstm_lm_600', 'wikitext-2')
        self.listenLoop()

    def listenLoop(self):
        f = self.f
        print("connected")
        while True:
            line = f.readline()
            f.write(str(self.lm.score(line)) + '\n')
            f.flush()

class NNLanguageModel:
    num_gpus = 0
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus else [mx.cpu()]
    log_interval = 200

    batch_size = 1 * len(context)
    lr = 20
    epochs = 3
    bptt = 1
    grad_clip = 0.25

    def __init__(self, awd_model_name, dataset_name):
        self.dataset_name = dataset_name
        self.train_dataset, self.test_dataset = [nlp.data.WikiText2(segment=segment, bos=None, eos='<eos>',
                            skip_empty=False) for segment in ['train', 'test']]
        self.vocab = nlp.Vocab(nlp.data.Counter(self.train_dataset[0]),
                            padding_token=None, bos_token=None)
        self.awd_model_name = awd_model_name
        self.awd_model, self.vocab = nlp.model.get_model(self.awd_model_name,
                                        dataset_name='wikitext-2',
                                        pretrained=True)
        print(self.awd_model)
        print(self.vocab)

    def detach(self, hidden):
        if isinstance(hidden, (tuple, list)):
            hidden = [self.detach(i) for i in hidden]
        else:
            hidden = hidden.detach()
        return hidden

    def score(self, sent):
        data = []
        data.append(sent.split())
        self.test_dataset._data = data
        test_data = self.test_dataset.bptt_batchify(self.vocab, self.bptt,
                                                    self.batch_size,
                                                    last_batch='discard')
        hidden = self.awd_model.begin_state(batch_size=self.batch_size,
                                            func=mx.nd.zeros,
                                            ctx=self.context[0])
        score = 0
        for i, (data, target) in enumerate(test_data):
            data = data.as_in_context(self.context[0])
            target = target.as_in_context(self.context[0])
            output, hidden = self.awd_model(data, hidden)
            lSoftmax = -mx.ndarray.log_softmax(output)
            lSoftmax = -mx.ndarray.log_softmax(output)
            score += lSoftmax.asnumpy().reshape(-1)[target[0,
                                                           0].asscalar().astype(int)]
            hidden = self.detach(hidden)
        return score

if __name__ == '__main__':
    lm_client = NNLanguageModelClient()
