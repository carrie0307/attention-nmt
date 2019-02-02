# -*- coding: utf-8 -*-
"""
 主函数
"""
print ("加载...")
from data_process import read_corpus,construct_wordid_dict, random_embedding, load_pickle_file
from model import NMT_test
import tensorflow as tf
import numpy as np
import pickle
import os, argparse, time, random
import warnings
warnings.filterwarnings('ignore')

# beam 0.39610317911773274
# greedy  0.42009784006625905

## hyperparameters
parser = argparse.ArgumentParser(description='RNN + Attention Translation Task')
parser.add_argument('--mode', type=str, default='infer', help='train / infer')
parser.add_argument('--embedding_dim', type=int, default=120, help='#dim of word embeddings')
parser.add_argument('--batch_size', type=int, default=50, help='#sample of each minibatch')
parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
parser.add_argument('--epoch_num', type=int, default=20, help='#epoch of training')
parser.add_argument('--num_units', type=int, default=120, help='#dim of hidden state')
parser.add_argument('--beam_width', type=int, default=0, help='# beas_width')
parser.add_argument('--model_path', type=str, default='', help='model_path')
args = parser.parse_args()
print ("加载数据...")
# 训练相关数据准备
train_cn_file, train_en_file = './data/cn.txt', './data/en.txt'
cn_word2id_dict,cn_id2word_dict = construct_wordid_dict(train_cn_file, ttype = 'src')
en_word2id_dict,en_id2word_dict = construct_wordid_dict(train_en_file, ttype = 'tgt')

cn_vocab_size, en_vocab_size = len(cn_word2id_dict), len(en_word2id_dict)
# cn_words_embedding = random_embedding(cn_vocab_size, args.embedding_dim)
# en_words_embedding = random_embedding(en_vocab_size, args.embedding_dim)
# with open('./word2vector_cn_embedding120.pkl', 'wb') as fw:
#     pickle.dump(cn_words_embedding, fw)
# with open('./word2vector_en_embedding120.pkl', 'wb') as fw:
#     pickle.dump(en_words_embedding, fw)
cn_words_embedding = load_pickle_file('./word2vector_cn_embedding120.pkl')
en_words_embedding = load_pickle_file('./word2vector_en_embedding120.pkl')

cn_train_data = read_corpus(train_cn_file, ttype='src')
en_train_data = read_corpus(train_en_file, ttype='tgt')

# 测试相关数据准备
dev_cn_file, dev_en_file = './data/cn_dev.txt', './data/en_dev.txt'
cn_dev_data = read_corpus(dev_cn_file, ttype='src')
en_dev_data = read_corpus(dev_en_file, ttype='tgt')

# 每次进行训练时生成一个时间戳
if args.mode == 'train':
    print ("\n训练模式...")
    timestamp = str(time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time())))
    if args.beam_width:
        timestamp = timestamp + "-beam"
    else:
        timestamp = timestamp + "-greedy"
else:
    pass
    print ("\n测试模式...")
    if args.beam_width:
        # timestamp = input("请输入要restore的模型生成时间戳,例如2018-12-12 19-58-25:")
        timestamp = "2018-12-19 16-34-52-beam"
    else:
        timestamp = "2018-12-19 16-18-47-greedy"
    print ("\n加载的模型是：", timestamp, '\n')

args.model_path = os.path.join('.', "model", timestamp + "/")
if not os.path.exists(args.model_path): os.makedirs(args.model_path)

if args.mode == 'train':

    model = NMT_test(args, cn_words_embedding, en_words_embedding, cn_word2id_dict, en_word2id_dict,en_id2word_dict)
    model.build_graph()
    model.train(cn_train_data, en_train_data)

elif args.mode == 'infer':

    args.model_path = tf.train.latest_checkpoint(args.model_path)
    model = NMT_test(args, cn_words_embedding, en_words_embedding, cn_word2id_dict, en_word2id_dict,en_id2word_dict)
    model.build_graph()
    print ("=============================")
    model.test(cn_dev_data, en_dev_data)




