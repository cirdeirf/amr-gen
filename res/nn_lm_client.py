import socket
import numpy as np
import gluonnlp as nlp
import mxnet as mx

import sys
import argparse

class NNLanguageModelClient:
    """Client to communicate with the amr-gen java project.

    Parameters
    ----------
    model_name: str
        The type of neural network language model to use. Options are
        'awd_lstm_lm_1150', 'awd_lstm_lm_600', 'standard_lstm_lm_1500',
        'standard_lstm_lm_650', 'standard_lstm_lm_200'
    """
    def __init__(self, model_name):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect(("localhost", 32000))
        self.f = self.s.makefile('rw')
        self.lm = NNLanguageModel(model_name)
        self.listenLoop()

    def listenLoop(self):
        """Listen to the java server for any requests to score sentences.

        The incoming strings have to be of the form '[n, s_0, ..., s_n-1]' where
        n is the batch size (number of sentences and s_i for 0<=i< n are the
        sentences to be scored.
        """
        f = self.f
        print("connected")
        while True:
            line = f.readline()
            line = line.split(" ", 1)
            f.write(str(self.lm.score(line[1], int(line[0]))) + '\n')
            f.flush()

class NNLanguageModel:
    """Neural network language model to score natural language sentences.

    The GluonNLP toolkit is used to load and execute different pretrained
    language models. Parts of this code have been adapted from GluonNLP's
    examples on language modeling.

    References
    ----------
        https://gluon-nlp.mxnet.io/
        https://gluon-nlp.mxnet.io/master/examples/language_model/language_model.html.

    Parameters
    ----------
    model_name : str
        The type of neural network language model to use. Options are
        'awd_lstm_lm_1150', 'awd_lstm_lm_600', 'standard_lstm_lm_1500',
        'standard_lstm_lm_650', 'standard_lstm_lm_200'.
    """
    def __init__(self, model_name):
        self.test_dataset = nlp.data.WikiText2(segment='test', bos='<bos>',
                                               eos='<eos>', skip_empty=False)        
        self.model_name = model_name
        # load the pretrained model
        self.model, self.vocab = nlp.model.get_model(self.model_name,
                                        dataset_name='wikitext-2',
                                        pretrained=True)
        print(self.model)
        print(self.vocab)
        self.num_gpus = 0
        self.context = [mx.gpu(i) for i in
                range(self.num_gpus)] if self.num_gpus else [mx.cpu()]
        self.bptt = 1

    def detach(self, hidden):
        """Detach gradients on states for truncated BPTT.
        """
        if isinstance(hidden, (tuple, list)):
            hidden = [self.detach(i) for i in hidden]
        else:
            hidden = hidden.detach()
        return hidden

    def score(self, sentences, batch_size):
        """Score a batch of sentences.

        Parameters
        ----------
        sentences : str
            String of concatenated sentences starting with a '<bos>' and ending
            with a '<eos>' token. To accomodate for different lengths, shorter
            sentences are padded with '<pad>'.

        batch_size : int
            Number of sentences contained in parameter 'sentences' and thus, the
            batch size.

        Returns
        -------
        scoreString : str
            String of space-separated scores (negative log_softmax) for each
            sentence.
        """
        batch_size = batch_size * len(self.context)
        data = []
        data.append(sentences.split())
        self.test_dataset._data = data
        test_data = self.test_dataset.bptt_batchify(self.vocab, self.bptt,
                                                    batch_size,
                                                    last_batch='discard')
        hidden = self.model.begin_state(batch_size=batch_size,
                                            func=mx.nd.zeros,
                                            ctx=self.context[0])
        score = mx.nd.array([0] * batch_size)
        eos_occured = [False] * batch_size
        # iterate over each word (data) and determine the probability for the
        # next word (target)
        for i, (data, target) in enumerate(test_data):
            # get/keep array on target device
            data = data.as_in_context(self.context[0])
            target = target.as_in_context(self.context[0])
            output, hidden = self.model(data, hidden)
            for j in range(batch_size):
                if (not eos_occured[j]):
                    # calculate score for each sentence by adding up the
                    # log_softmax for each word in the sentence
                    score[j] += mx.ndarray.log_softmax(output)[0, j, target[0, j]]
                if (target[0, j].asscalar().astype(int) == 1):
                    # stop if the token '<eos>' has occurred for a sentence
                    eos_occured[j] = True
            hidden = self.detach(hidden)
        scoreString = ' '.join(str(s.asscalar()) for s in score)
        return scoreString

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Listen to amr-gen and score \
            sentences using a neural network language model.')
    parser.add_argument("-m", "--model", type = str, help = "Choose the neural \
        network model to use. Possible choices: awd_lstm_lm_1150, \
        awd_lstm_lm_600, standard_lstm_lm_1500, standard_lstm_lm_650, \
        standard_lstm_lm_200 (default: awd_lstm_lm_600).", default =
        'awd_lstm_lm_600')
    args = parser.parse_args()
    lm_client = NNLanguageModelClient(args.model)
