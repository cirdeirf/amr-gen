#  import warnings
#  warnings.filterwarnings('ignore')

import mxnet as mx
from mxnet import gluon, autograd, ndarray
from mxnet.gluon.utils import download

import gluonnlp as nlp

import numpy as np
import time

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
            lSoftmax = -ndarray.log_softmax(output)
            lSoftmax = -ndarray.log_softmax(output)
            score += lSoftmax.asnumpy().reshape(-1)[target[0,
                                                           0].asscalar().astype(int)]
            hidden = self.detach(hidden)
        print(type(score))
        return score

#  lm = NNLanguageModel('awd_lstm_lm_600', 'wikitext-2')
#  sent = 'I that that this is a disgraceful situation that should not be tolerated'
#  print(sent)
#  startTime = time.time()
#  print(lm.score(sent))
#  elapsedTime = time.time() - startTime
#  print('time', elapsedTime)
#  print()
