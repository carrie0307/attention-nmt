# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
import sys
from assist_func import label_smoothing, multihead_attention, positional_encoding, normalize, embedding, feedforward
from data_process import batch_yield, pad_sequences, sentence2word
from bleu import calculate_bleu
import warnings
warnings.filterwarnings('ignore')


class Graph():
    
    def __init__(self, args, embedding_cn, embedding_en, cn_word2id_dict, en_word2id_dict,en_id2word_dict):         
        
        self.is_training = args.is_training
        self.epoch_num = args.epoch_num
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.dropout_rate = args.dropout_rate
        self.num_units = args.num_units

        # 6个blocks
        self.num_blocks = args.num_blocks
        self.num_heads = args.num_heads
        self.sinusoid = False

        # 原文和译文词向量(lookup_table)
        self.embedding_cn, self.embedding_en = embedding_cn, embedding_en
        self.cn_word2id_dict = cn_word2id_dict
        self.en_word2id_dict = en_word2id_dict
        self.en_id2word_dict = en_id2word_dict
        self.model_path = args.model_path


    def build_graph(self):
        
        self.add_placeholders()
        self.build_encoder()
        self.build_decoder()
        self.loss_op()
        self.init_op()


    def init_op(self):
        self.init_op = tf.global_variables_initializer()


    def add_placeholders(self):

        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        # self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
        # self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')


    def build_encoder(self):
        
        with tf.variable_scope("encoder") as scope:

            # 查询词向量
            self.encoder_emb_in = embedding(lookup_table = self.embedding_cn,
                                            inputs = self.encoder_inputs, 
                                            num_units = self.num_units,
                                            scale = True,
                                            scope = "enc_embed")            
            # 位置信息编码
            if self.sinusoid:
                inputs = self.encoder_inputs
                self.encoder_emb_in = positional_encoding(inputs = inputs,
                                                          num_units = self.num_units,
                                                          zeor_pad = False,
                                                          scale = False,
                                                          scope = "enc_pe")
            else:
                # 将inputs变为没个词位置的id
                inputs = tf.tile(tf.expand_dims(tf.range(tf.shape(self.encoder_inputs)[1]), 0), [tf.shape(self.encoder_inputs)[0], 1])
                self.encoder_emb_in += embedding( lookup_table = self.embedding_cn,
                                                  inputs = inputs, 
                                                  num_units = self.num_units,
                                                  scale=False,
                                                  scope="enc_pe")

            # Dropout
            self.encoder_emb_in = tf.layers.dropout(inputs = self.encoder_emb_in, 
                                                    rate = self.dropout_rate, 
                                                    training=tf.convert_to_tensor(self.is_training))

            # Blocks
            # 编码器由6个相同的层堆叠在一起，每一层包括一个多头的自注意机制和一个全连接前馈网络 (Norm的操作包含在这两部分里面进行)
            for i in range(self.num_blocks):
                """
                每个block都是 多头Attention + Feedworward前馈
                """
                with tf.variable_scope("num_blocks_{}".format(i)):

                    ### 多头Attention
                    # 多头Attentiond是基于self_attention, 因此Q,K,V相同都是self.encoder_emb_in
                    self.encoder_emb_in = multihead_attention(queries = self.encoder_emb_in,
                                                              keys = self.encoder_emb_in,
                                                              num_units = self.num_units,
                                                              dropout_rate = self.dropout_rate,
                                                              is_training = self.is_training,
                                                              causality = False)
                    # 多头Attention的normalize操作包含在了multihead_attention函数的最后一步进行

                    ## 前馈神经网
                    self.encoder_emb_in = feedforward(inputs = self.encoder_emb_in,
                                                      num_units = [4 * self.num_units, self.num_units])
                    # 前馈网络的normalize操作包含在了feedforward函数的最后一步进行



    def build_decoder(self):
        
        with tf.variable_scope("decoder"):

            # 去掉句末的<EOS>
            ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
            # 添加上句首的<GO>
            self.decoder_inputs = tf.concat([tf.fill([self.batch_size, 1], self.en_word2id_dict['<GO>']), ending], 1)

            ## 查询词向量
            self.decoder_emb_in = embedding(lookup_table = self.embedding_en,
                                            inputs = self.decoder_inputs, 
                                            num_units = self.num_units,
                                            scale=True,
                                            scope="dec_embed")

            # 位置信息编码
            if self.sinusoid:
                inputs = self.decoder_inputs
                self.decoder_emb_in = positional_encoding(inputs = inputs,
                                                          num_units = self.num_units,
                                                          zeor_pad = False,
                                                          scale = False,
                                                          scope = "dec_pe")
            else:
                inputs = tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1])
                self.decoder_emb_in += embedding( lookup_table = self.embedding_en,
                                                  inputs = inputs, 
                                                  num_units = self.num_units,
                                                  scale=False,
                                                  scope="dec_pe")

            ## Dropout
            self.decoder_emb_in = tf.layers.dropout(inputs = self.decoder_emb_in,
                                                    rate = self.dropout_rate,
                                                    training = tf.convert_to_tensor(self.is_training))

            ## Blocks
            # 解码器由6个相同的层堆叠在一起，每一层包括一个多头的自注意机制和一个全连接前馈网络 (Norm的操作包含在这两部分里面进行)
            for i in range(self.num_blocks):

                with tf.variable_scope("num_blocks_{}".format(i)):

                    # 多头Attentiond是基于self_attention, 因此Q,K,V相同都是self.decoder_emb_in
                    ## 多头 self attention
                    # 注意这里causality=TRue, 需要进行mask
                    self.decoder_emb_in = multihead_attention(queries = self.decoder_emb_in,
                                                              keys = self.decoder_emb_in,
                                                              num_units = self.num_units,
                                                              num_heads = self.num_heads,
                                                              dropout_rate = self.dropout_rate,
                                                              is_training = self.is_training,
                                                              causality = True,
                                                              scope = "self_attention")
                    # 多头Attention的normalize操作包含在了multihead_attention函数的最后一步进行

                    """
                    编码可以并行计算，一次性全部encoding出来，但解码不是一次把所有序列解出来的，而是像rnn一样一个一个解出来的 ，
                    因为要用上一个位置的输入当作attention的query
                    """

                    # 解码器的self attention之后跟了一个和编码器输出作为keys的attention，从而将编码器和解码器联系起来。
                    # 这里causality=False, 不用进行mask
                    # https://blog.csdn.net/bobobe/article/details/82629393
                    # 得到编码层的key和value以及解码层的query后，下面就是模仿vanilla attention，利用key和value以及query再做最后一个attention。得到每个位置的输出
                    self.decoder_emb_in = multihead_attention(queries = self.decoder_emb_in,
                                                              keys = self.encoder_emb_in,
                                                              num_units = self.num_units,
                                                              num_heads = self.num_heads,
                                                              dropout_rate = self.dropout_rate,
                                                              is_training = self.is_training,
                                                              causality = False,
                                                              scope = "vanilla_attention")

                    ## 前馈网络
                    self.decoder_emb_in = feedforward(inputs=self.decoder_emb_in,
                                                      num_units=[4 * self.num_units,self.num_units])
                    # 前馈网络的normalize操作包含在了feedforward函数的最后一步进行
                    # 最终输出self.decoder_emb_in的shape为[N,T,self.num_units]


    def loss_op(self):

        self.logits = tf.layers.dense(self.decoder_emb_in, len(self.en_word2id_dict))
        self.preds = tf.to_int32(tf.arg_max(self.logits, dimension = -1))
        self.istarget = tf.to_float(tf.not_equal(self.decoder_targets, 0))
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.decoder_targets)) * self.istarget) / (tf.reduce_sum(self.istarget))
        # tf.summary.scalar('acc', self.acc)

        if self.is_training:
            ## Loss
            self.decoder_tartets_smoothed = label_smoothing(tf.one_hot(self.decoder_targets, depth=len(self.en_word2id_dict)))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                                labels=self.decoder_tartets_smoothed)
            self.train_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))

            ## Training Schema
            self.global_step = tf.Variable(0, name = 'global_step', trainable = False)
            self.optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.train_op=self.optimizer.minimize(self.train_loss, global_step=self.global_step)

    def train(self, cn_train_data, en_train_data):

        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            
            sess.run(self.init_op)
            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess, cn_train_data, en_train_data, epoch, saver)

    def run_one_epoch(self, sess, cn_train_data, en_train_data, epoch, saver):

        num_batches = (len(cn_train_data) + self.batch_size - 1) // self.batch_size
        
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # batch的时候进行id操作
        train_batches = batch_yield(cn_train_data, en_train_data, self.batch_size,
                                    self.cn_word2id_dict, self.en_word2id_dict)

        for step, (encoder_input ,decoder_input) in enumerate(train_batches):

            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            
            # 进行pad操作
            # 得到id化句子padding后的结果，以及每个句子真实长度
            encoder_seq_list, encoder_len_list,decoder_seq_list, decoder_len_list = pad_sequences(encoder_input,
                                                                                                  decoder_input, 
                                                                                                  self.cn_word2id_dict, 
                                                                                                  self.en_word2id_dict,
                                                                                                  pad_mark='<PAD>')
            infer_init_decoder_seq_list = np.ones
            feed_dict = { self.encoder_inputs: encoder_seq_list,
                          self.encoder_inputs_length: encoder_len_list,
                          # self.encoder_inputs_maxlen: encoder_seq_list[0],

                          self.decoder_targets: decoder_seq_list,
                          # self.decoder_targets_length: decoder_len_list,
                          # self.decoder_targets_maxlen: decoder_seq_list[0]
                         }

            _, loss_train = sess.run([self.train_op, self.train_loss], feed_dict=feed_dict)


            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:

                print ('{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, 
                                                                                   step + 1,loss_train, step_num))

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step = step_num)


    def test(self, cn_sent, en_sent):

        max_length = 29
        eos_id = self.en_word2id_dict['<PAD>']
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, self.model_path)

            batches = batch_yield(cn_sent, en_sent, self.batch_size, self.cn_word2id_dict, self.en_word2id_dict)

            all_candidates = []
            for i, (src, tgt) in enumerate(batches):

                print ("{} / 26 batches , wait please ...".format(str(i + 1)))
                encoder_seq_list, encoder_len_list, decoder_seq_list, decoder_len_list = pad_sequences(src, tgt,
                                                                                                       self.cn_word2id_dict,
                                                                                                       self.en_word2id_dict,
                                                                                                       pad_mark='<PAD>')
                # infer_decoder_seq_list = np.ones([self.batch_size, max_length])
                # print ("deocder_length: ", decoder_len_list)
                # print ("decoder pad length: ", [len(i) for i in decoder_seq_list])
                feed_dict = {self.encoder_inputs: encoder_seq_list,
                             self.encoder_inputs_length: encoder_len_list,
                             # 初始化输入以pad作为输入
                             self.decoder_targets: np.ones([self.batch_size, max_length]) * self.en_word2id_dict['<PAD>'],
                             }

                # 用predicts来获取真正的结果
                predicts = np.zeros((self.batch_size, max_length), np.int32)

                # 在当前batch的句子从第一个词开始，逐词预测，从而后一个词预测的时候就可以利用前面的信息来解码
                for i in range(max_length):
                    # _predicts[array([],[], ..., [])]
                    _predicts = sess.run([self.preds], feed_dict=feed_dict)
                    # print (_predicts[0][:, i])
                    # _predicts[0] array([], [],...)
                    predicts[:,i] = _predicts[0][:,i]
                # predicts是最终这一个batch内的预测结果

                # 将id化的翻译结果转换为单词
                for sent in predicts:
                    sent = list(sent)
                    if eos_id in sent:
                        sent = sent[:sent.index(eos_id)]
                    sent_word = sentence2word(sent, self.en_id2word_dict)
                    all_candidates.append(sent_word)

        calculate_bleu(all_candidates, en_sent)


