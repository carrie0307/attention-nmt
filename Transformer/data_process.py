# -*- coding: utf-8 -*-
import collections
import numpy as np
import random
import pickle

def construct_wordid_dict(filename, ttype, min_count = 1):

    """
    读取数据构造word与id互相转换的字典
    """

    word_counts = collections.Counter()
    word2id = {}

    with open(filename, 'r', encoding = 'utf8') as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()
            word_counts.update(line.split())
    
    # print (word_counts.most_common())
    vocabulary_inv = [x[0] for x in word_counts.most_common() if x[1] >= min_count]
    if ttype == 'tgt':
        vocabulary_inv.append('<GO>')
        vocabulary_inv.append('<EOS>')
    vocabulary_inv.append('<PAD>')
    word2id_dict = {x:i for i,x in enumerate(vocabulary_inv)}
    id2word_dict = {i:x for i,x in enumerate(vocabulary_inv)}
    return word2id_dict,id2word_dict



def read_corpus(filename, ttype):
    
    """
    读取数据，将每个句子用一个以词分割的列表表示
    """
    corpus = []
    with open(filename, 'r', encoding = 'utf8') as f:

        lines = f.readlines()
        for line in lines:
            line = line.strip()
            sent = line.split()
            if ttype == 'tgt':
                sent.append('<EOS>')
            corpus.append(sent)

    return corpus


def random_embedding(vocab_size, embedding_dim):
    """
    随机生成词向量
    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    embedding_mat = np.float32(embedding_mat)
    return embedding_mat


def sentence2id(sent, wordid_dict):
    """
    把句子转化成id
    :param sent:[word1, word2, ...]
    :return idsent: [word1_id, word2_id, ...]
    """
    # QUESTION: GO,EOS的处理
    id_sent = []
    # 除去GO,EOS和PAD
    length = len(wordid_dict) - 3
    for word in sent:
        id_sent.append(wordid_dict.get(word, random.randint(1,length)))
        # id_sent.append(wordid_dict[word])
    return id_sent


def batch_yield(cn_data, en_data, batch_size, cn_word2id_dict, en_word2id_dict):
    
    """
    进行batch的分割，返回按batch_size分割好后的数据
    """

    if en_data:
        cn_sents, en_sents = [],[]
        data = zip(cn_data, en_data)

        for cn_sent, en_sent in data:
            cn_id_sent = sentence2id(cn_sent, cn_word2id_dict)
            en_id_sent = sentence2id(en_sent, en_word2id_dict)

            if len(cn_sents) == batch_size and len(en_sents) == batch_size:
                yield cn_sents, en_sents
                cn_sents, en_sents = [],[]

            cn_sents.append(cn_id_sent)
            en_sents.append(en_id_sent)

        if len(cn_sents) != 0:
            yield cn_sents, en_sents

def pad_sequences(encoder_inputs, decoder_inputs, cn_word2id_dict, en_word2id_dict, pad_mark= '<PAD>'):

    encoder_max_len = max(map(lambda x:len(x), encoder_inputs))
    decoder_max_len = max(map(lambda x:len(x), decoder_inputs))

    encoder_seq_list, encoder_len_list = [], []
    decoder_seq_list, decoder_len_list = [], []
    data = zip(encoder_inputs, decoder_inputs)

    encoder_pad_id = cn_word2id_dict[pad_mark]
    decoder_pad_id = en_word2id_dict[pad_mark]

    for encoder_seq, decoder_seq in data:

        encoder_seq_ = encoder_seq[:encoder_max_len] + [encoder_pad_id] * max(encoder_max_len - len(encoder_seq), 0)
        encoder_seq_list.append(encoder_seq_)
        encoder_len_list.append(min(len(encoder_seq), encoder_max_len))

        decoder_seq_ = decoder_seq[:decoder_max_len] + [decoder_pad_id] * max(decoder_max_len - len(decoder_seq), 0)
        decoder_seq_list.append(decoder_seq_)
        decoder_len_list.append(min(len(decoder_seq), decoder_max_len))

    return encoder_seq_list, encoder_len_list,decoder_seq_list, decoder_len_list


def sentence2word(sent, wordid_dict):
    """
    将译文的id转化为word
    :param sent:
    :param wordid_dict:
    :return:
    """
    word_sent = []
    for word_id in sent:
        word_sent.append(wordid_dict[word_id])
    return word_sent


def load_pickle_file(filename):
    with open(filename, 'rb') as fr:
        content = pickle.load(fr)
    return content


if __name__ == '__main__':
    # construct_wordid_dict('./data/en-test.txt', min_count = 1)

    cn_word2id_dict,cn_id2word_dict = construct_wordid_dict('./data/cn-test.txt')
    en_word2id_dict,en_id2word_dict = construct_wordid_dict('./data/en-test.txt')

    cn_vocab_size, en_vocab_size = len(cn_word2id_dict), len(en_word2id_dict)

    train_cn_data = read_corpus('./data/cn-test.txt')
    train_en_data = read_corpus('./data/en-test.txt')

    data = batch_yield(train_cn_data, train_en_data, 20, cn_word2id_dict, en_word2id_dict)