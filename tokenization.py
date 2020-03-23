#!/usr/bin/env python
# -*-encoding=utf-8-*-
import jieba
import collections
import tensorflow as tf
import six


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def load_word_vocab(word_vocab_file):
    vocab = collections.OrderedDict()
    index = 0

    skip_head = True
    with tf.gfile.GFile(word_vocab_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            if skip_head:
                skip_head = False
                continue
            token = token.strip().split(' ')[0]
            vocab[token] = index
            index += 1
    return vocab


def load_stop_words(stop_words_file):
    stop_words = set()
    index = 0
    with tf.gfile.GFile(stop_words_file, "r") as reader:
        while True:
            token = convert_to_unicode(reader.readline())
            if not token:
                break
            token = token.strip()
            stop_words.add(token)
    return stop_words


def convert_by_vocab(vocab, items):
    """Converts a sequence of [tokens|ids] using the vocab."""
    output = []
    for item in items:
        output.append(vocab[item])
    return output


class Tokenizer(object):
    def __init__(self, word2vec_file, stop_words_file=None):
        self.vocab = load_word_vocab(word2vec_file)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        if stop_words_file is None:
            self.stop_words = set()
        else:
            self.stop_words = load_stop_words(stop_words_file)

    def tokenize(self,text):
        split_tokens = []
        text = text.strip()
        words = jieba.cut(text)
        useful_words = self.filter_stop_words(words)
        return useful_words

    def filter_stop_words(self,words_list):
        useful_words = []
        for w in words_list:
            if w in self.stop_words:
                continue
            useful_words.append(w)
        return useful_words

    def convert_tokens_to_ids(self, tokens):
        return convert_by_vocab(self.vocab, tokens)

    def convert_ids_to_tokens(self, ids):
        return convert_by_vocab(self.inv_vocab, ids)


