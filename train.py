import os
import json
import numpy as np
from gensim.models.word2vec import LineSentence
from gensim.models import FastText, Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

import logging
logging.basicConfig(level=logging.INFO, force = True)
logger = logging.getLogger()
logger.info("Logging initialized")


class Callback(CallbackAny2Vec):
    def __init__(self, loss_list):
        self.epoch = 0
        self.loss_list = loss_list

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.loss_list.append(loss)
        logger.info('Loss after epoch {}:{}'.format(self.epoch, loss))
        model.running_training_loss = 0.0
        self.epoch = self.epoch + 1


LOSSES_FILENAME = 'losses.json'


def save_losses(name, losses):
    losses_mapping = {}
    if os.path.exists(LOSSES_FILENAME):
        with open(LOSSES_FILENAME) as f:
            losses_mapping = json.load(f)
    losses_mapping[name] = losses
    with open(LOSSES_FILENAME, 'w') as f:
        json.dump(losses_mapping, f, ensure_ascii=False, indent=0, sort_keys=True)


def get_losses(name):
    result = []
    if os.path.exists(LOSSES_FILENAME):
        with open(LOSSES_FILENAME) as f:
            losses_mapping = json.load(f)
        if name in losses_mapping:
            result = losses_mapping[name]
    return result


def train(arch, sg, vector_size, window, min_count, ns_exponent, sample, epochs, processed_filename, model_descr=''):
    """
    :param arch: 'w2v' (for word2vec) or 'ft' (for fasttext)
    :param sg: 0 or 1, same as in gensim
    :param vector_size: dimension of embeddings, same as in gensim
    :param window: size of context during training, same as in gensim
    :param min_count: minimum frequency for a word to be included, same as in gensim
    :param ns_exponent: exponent for negative sampling, same as in gensim
    :param sample: threshold for downsampling words, same as in gensim,
    :param epochs: amount of epochs to train, same as in gensim
    :param processed_filename: filename of processed corpus
    :param model_descr: additional description for model to be included along standard specs,
    shouldn't contain hyphen ('-')
    :return: list of losses for each epoch
    """
    ns_exp_fmt = np.format_float_positional(ns_exponent, trim="-")
    sample_fmt = np.format_float_positional(sample, trim="-")
    name = f'{arch}-sg{sg}-d{vector_size}-w{window}-min{min_count}-nse{ns_exp_fmt}-smp{sample_fmt}-ep{epochs}'
    if model_descr:
        name = name + f'-descr_{model_descr}'
    logger.info(f'Model name: {name}')
    if os.path.exists(name + '.model') and os.path.getsize(name + '.model') > 0:
        losses = get_losses(name)
        if losses:
            logger.info('Model already exists, skipping training, returning previously calculated losses')
            return losses

    logger.info('Training model from scratch...')
    losses = []
    if arch == 'w2v':
        model = Word2Vec(sg=sg, vector_size=vector_size, window=window, min_count=min_count, workers=5,
                         ns_exponent=ns_exponent, sample=sample)
    elif arch == 'ft':
        model = FastText(sg=sg, vector_size=vector_size, window=window, min_count=min_count, workers=5,
                         ns_exponent=ns_exponent, sample=sample)
    else:
        raise Exception(f'Unknown architecture: {arch}')
    sentences = LineSentence(processed_filename)
    model.build_vocab(sentences, progress_per=5000000)
    # we override alpha with small values, since default values result in poor train performance
    model.train(sentences, epochs=epochs, start_alpha=0.0001, end_alpha=0.00001, total_examples=model.corpus_count,
                total_words=model.corpus_total_words, compute_loss=True, report_delay=300,
                callbacks=[Callback(losses)])

    model.save(f'{name}.model')
    model.wv.save_word2vec_format(f'{name}.vectors')
    save_losses(name, losses)
    return losses


if __name__ == "__main__":
    train(arch='ft', sg=0, vector_size=100, window=3, min_count=10, ns_exponent=0.75, sample=0.001, epochs=100,
          processed_filename='processed-corpus.txt')
    train(arch='w2v', sg=0, vector_size=100, window=3, min_count=10, ns_exponent=0.75, sample=0.001, epochs=100,
          processed_filename='processed-corpus-only-nouns.txt', model_descr='only_nouns')
    train(arch='w2v', sg=0, vector_size=100, window=3, min_count=10, ns_exponent=0.1, sample=0.001, epochs=100,
          processed_filename='processed-corpus.txt')
    train(arch='w2v', sg=0, vector_size=100, window=3, min_count=10, ns_exponent=0.1, sample=0.00001, epochs=100,
          processed_filename='processed-corpus.txt')
    train(arch='w2v', sg=0, vector_size=100, window=3, min_count=10, ns_exponent=0.1, sample=0.00001, epochs=500,
          processed_filename='processed-corpus.txt')
    train(arch='w2v', sg=0, vector_size=100, window=3, min_count=10, ns_exponent=0.75, sample=0.001, epochs=100,
          processed_filename='processed-corpus.txt')
    train(arch='w2v', sg=0, vector_size=100, window=3, min_count=10, ns_exponent=0.75, sample=0.001, epochs=200,
          processed_filename='processed-corpus.txt')
    train(arch='w2v', sg=0, vector_size=100, window=3, min_count=10, ns_exponent=0.75, sample=0.001, epochs=500,
          processed_filename='processed-corpus.txt')
    train(arch='w2v', sg=0, vector_size=100, window=3, min_count=10, ns_exponent=0.75, sample=0.0001, epochs=100,
          processed_filename='processed-corpus.txt')
    train(arch='w2v', sg=0, vector_size=100, window=3, min_count=10, ns_exponent=0.75, sample=0.00001, epochs=100,
          processed_filename='processed-corpus.txt')
    train(arch='w2v', sg=0, vector_size=100, window=3, min_count=30, ns_exponent=0, sample=10000, epochs=300,
          processed_filename='processed-corpus.txt')
    train(arch='w2v', sg=0, vector_size=100, window=3, min_count=100, ns_exponent=0, sample=10000, epochs=200,
          processed_filename='processed-corpus.txt')
    train(arch='w2v', sg=0, vector_size=100, window=3, min_count=100, ns_exponent=0, sample=10000, epochs=200,
          processed_filename='processed-corpus-only-nouns.txt', model_descr='only_nouns')
    train(arch='w2v', sg=0, vector_size=200, window=3, min_count=10, ns_exponent=0.75, sample=0.001, epochs=100,
          processed_filename='processed-corpus.txt')
    train(arch='w2v', sg=1, vector_size=100, window=3, min_count=10, ns_exponent=0.75, sample=0.001, epochs=100,
          processed_filename='processed-corpus.txt')
    train(arch='w2v', sg=0, vector_size=100, window=4, min_count=100, ns_exponent=0, sample=10000, epochs=200,
          processed_filename='processed-corpus.txt')



