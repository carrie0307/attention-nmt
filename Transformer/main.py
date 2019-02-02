# -*- coding: utf-8 -*-
from data_process import read_corpus,construct_wordid_dict, random_embedding, load_pickle_file
from model import Graph
import tensorflow as tf
import numpy as np
import os, argparse, time, random
import warnings
warnings.filterwarnings('ignore')

# bleu: 0.6116908849062525
# 0.55

## hyperparameters
parser = argparse.ArgumentParser(description='Transformer Translation Task')
# parser.add_argument('--is_training', type=bool, default=False, help='True / False')
parser.add_argument('--is_training', type=int, default=0, help='1 / 0')
parser.add_argument('--embedding_dim', type=int, default=120, help='#dim of word embeddings')
parser.add_argument('--batch_size', type=int, default=50, help='#sample of each minibatch')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--dropout_rate', type=float, default='0.1', help='Dropout rate')
parser.add_argument('--epoch_num', type=int, default=20, help='#epoch of training')
parser.add_argument('--num_units', type=int, default=120, help='#dim of hidden state')
parser.add_argument('--num_blocks', type=int, default= 6, help='num of encoder/decoder blocks')
parser.add_argument('--num_heads', type=int, default=8, help='')
parser.add_argument('--model_path', type=str, default='', help='model_path')
args = parser.parse_args()

# 训练相关数据准备
train_cn_file, train_en_file = './data/cn.txt', './data/en.txt'
cn_word2id_dict,cn_id2word_dict = construct_wordid_dict(train_cn_file,ttype = 'src')
en_word2id_dict,en_id2word_dict = construct_wordid_dict(train_en_file, ttype = 'tgt')

cn_vocab_size, en_vocab_size = len(cn_word2id_dict), len(en_word2id_dict)
# print(cn_vocab_size, en_vocab_size)
cn_words_embedding = load_pickle_file('./word2vector_cn_embedding120.pkl')
en_words_embedding = load_pickle_file('./word2vector_en_embedding120.pkl')

cn_train_data = read_corpus(train_cn_file, ttype='src')
en_train_data = read_corpus(train_en_file, ttype='tgt')

# 测试相关数据准备
dev_cn_file, dev_en_file = './data/cn_dev.txt', './data/en_dev.txt'
cn_dev_data = read_corpus(dev_cn_file, ttype='src')
en_dev_data = read_corpus(dev_en_file, ttype='tgt')

# 每次进行训练时生成一个时间戳
if args.is_training:
    print ("\n训练模式 .... \n")
    args.is_training = True
    timestamp = str(time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())))
else:
    pass
    print ("\n测试模式...")
    # timestamp = input("请输入要restore的模型生成时间戳,例如2018-12-12 19-58-25:")
    timestamp = "2018-12-19 17-39-12"
    # timestamp = "2018-12-19 17-03-59"
    print ("\n当前加载的模型是: ", timestamp, '\n')
    args.is_training = False

args.model_path = os.path.join('.', "model", timestamp + "/")
# args.model_path = os.path.join('.', "model", 'test' + "/")
if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

if args.is_training:

    model = Graph(args, cn_words_embedding, en_words_embedding, cn_word2id_dict, en_word2id_dict,en_id2word_dict)
    model.build_graph()
    model.train(cn_train_data, en_train_data)

else:

    args.model_path = tf.train.latest_checkpoint(args.model_path)
    model = Graph(args, cn_words_embedding, en_words_embedding, cn_word2id_dict, en_word2id_dict,en_id2word_dict)
    model.build_graph()
    print ("=============================")
    model.test(cn_dev_data, en_dev_data)




