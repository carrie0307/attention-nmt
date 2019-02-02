# -*- coding: utf-8 -*-
"""
    sRNN + Attention模型
"""
import tensorflow as tf 
import time
import numpy as np
from collections import Counter
from data_process import batch_yield, pad_sequences, sentence2word
from tensorflow.python.layers import core as layers_core
from bleu import calculate_bleu
import sys

class NMT_test(object):

    def __init__(self, args, embedding_cn, embedding_en, cn_word2id_dict, en_word2id_dict,en_id2word_dict):

        self.mode = args.mode
        self.epoch_num = args.epoch_num
        self.batch_size = args.batch_size
        self.num_units = args.num_units
        self.beam_width = args.beam_width
        self.learning_rate = args.learning_rate
        self.cn_word2id_dict = cn_word2id_dict
        self.en_word2id_dict = en_word2id_dict
        self.en_id2word_dict = en_id2word_dict
        self.model_path = args.model_path

        # 原文和译文词向量，需要进行lookup
        self.embedding_cn, self.embedding_en = embedding_cn, embedding_en


    def build_graph(self):
        """
           搭建整个模型
        """
        self.add_palceholders()
        # tf.nn.embedding_lookup分别在encoder()和decoder()中进行
        self.build_encoder()
        self.build_decoder()
        self.loss_op()
        self.init_op()


    def add_palceholders(self):
        
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')

    def build_encoder(self):
        """
        定义encoder
        :return:
        """

        with tf.variable_scope('encoder') as scope:
            
            # 得到所有待训练数据的词向量化的所有待训练数据
            self.encoder_emb_in = tf.nn.embedding_lookup(self.embedding_cn, self.encoder_inputs)
            

            encoder_cell = tf.contrib.rnn.BasicRNNCell(self.num_units) 
            # sequence_length是每个句子实际的长度
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(cell = encoder_cell, 
                                                                         inputs = self.encoder_emb_in,
                                                                         sequence_length = self.encoder_inputs_length,
                                                                         dtype = tf.float32)


    def build_decoder(self):
        """
        定义decoder
        :return:
        """
        with tf.variable_scope('decoder') as scope:

            # 构建具有attention的decode_cell
            decoder_cell, decoder_initial_state = self._build_decoder_cell()
            # QUESTION: projection_layer, a dense matrix to turn the top hidden states to logit vectors of dimension V
            projection_layer = layers_core.Dense(len(self.en_word2id_dict), use_bias=False)
            maximum_iterations = self.max_target_sequence_length

            if self.mode == 'train':

                # 去掉句末的<EOF>
                ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
                # 添加上句首的<GO>
                decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.en_word2id_dict['<GO>']), ending], 1)
                # 通过lookup表获取词向量
                self.decoder_emb_in = tf.nn.embedding_lookup(self.embedding_en, decoder_input)
                # Helper
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs = self.decoder_emb_in, 
                                                                    sequence_length = self.decoder_targets_length)
                # Decoder
                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell = decoder_cell, 
                                                                    helper = training_helper, 
                                                                    initial_state = decoder_initial_state, 
                                                                    output_layer = projection_layer)
                # Dynamic decoding
                decoder_outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                                                            scope = scope)

                # QUESTION: 这里不太确定
                # decoder_outputs是一个namedtuple，里面包含两项(rnn_outputs, sample_id)
                # rnn_outputs: [batch_size, decoder_targets_length, vocab_size]
                # sample_d: [batch_size] 保存最后的编码结果，可以表示最后的答案
                self.decoder_sample_id = decoder_outputs.sample_id
                # 用交叉熵计算应该就是去decoder_outputs.run_output作为logits
                self.logits = decoder_outputs.rnn_output
                self.decoder_final_context_state = final_context_state

            elif self.mode == 'infer':

                start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.en_word2id_dict['<GO>']
                end_token = self.en_word2id_dict['<EOS>']

                if self.beam_width:

                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell, embedding=self.embedding_en,
                                                                             start_tokens=start_tokens, end_token=end_token,
                                                                             initial_state=decoder_initial_state,
                                                                             beam_width=self.beam_width,
                                                                             output_layer=projection_layer)
                else:
                    # 不用BeasSearch,则用GreedyEmbedding
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=self.embedding_en,
                                                                      start_tokens=start_tokens, 
                                                                      end_token=end_token)
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                                        helper=helper,
                                                                        initial_state=decoder_initial_state,
                                                                        output_layer=projection_layer)
                decoder_outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder = inference_decoder,
                                                                                            maximum_iterations = maximum_iterations,
                                                                                            scope = scope)
                # 使用Beam_search时，decoder_outputs里面包含两项(predicted_ids, beam_search_decoder_output)
                # predicted_ids: [batch_size, decoder_targets_length, beam_size]保存输出结果
                # beam_search_decoder_output: BeamSearchDecoderOutput instance namedtuple(scores, predicted_ids, parent_ids)
                # 所以只需返回predicted_ids或sample_id即可翻译成最终的结果
                if self.beam_width:
                    self.logits = tf.no_op()
                    self.predicted_sample_id = decoder_outputs.predicted_ids
                else:
                    self.logits = decoder_outputs.rnn_output
                    self.predicted_sample_id = decoder_outputs.sample_id


    def _build_decoder_cell(self):

        beam_width = self.beam_width
        dtype = tf.float32
        # 由于tf.nn.dynamic_rnn的time_major=False, 故RNN的最后输出可直接作为memory
        memory, encoder_state, encoder_inputs_length = self.encoder_outputs, self.encoder_state, self.encoder_inputs_length

        if self.mode == 'infer' and beam_width > 0:

            memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=beam_width)
            encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)
            encoder_inputs_length = tf.contrib.seq2seq.tile_batch(encoder_inputs_length, multiplier=beam_width)
            # 因为进行了复制，因此修改batch_size
            batch_size = self.batch_size * beam_width
        else:
            # 不使用beam的情况
            batch_size = self.batch_size
        
        # LuongAttention: align(h,s)
        # Bahdanauattention : eij =a(si−1,hj) a相当于对齐函数
        # memory: the output of an RNN encoder. This tensor should be shaped [batch_size, max_time, ...]
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(self.num_units, memory = memory,
                                                                memory_sequence_length=encoder_inputs_length)
        basic_cell = tf.contrib.rnn.BasicRNNCell(self.num_units)
        alignment_history = (self.mode == 'infer' and beam_width == 0)
        # alignment_history: Python boolean, whether to store alignment history from all time steps in the final output state
        cell = tf.contrib.seq2seq.AttentionWrapper(basic_cell, attention_mechanism,
                                                   attention_layer_size=self.num_units,
                                                   alignment_history=alignment_history,
                                                   output_attention=True,
                                                   name='attention')
        # 定义decoder阶段的初始化状态，直接使用encoder阶段的最后一个隐层状态进行赋值
        decoder_initial_state = cell.zero_state(batch_size, dtype).clone(cell_state=encoder_state)
        return cell, decoder_initial_state


    def loss_op(self):
        
        if self.mode == 'train':
            # labels应该是decode的结果
            # target_output
            max_times = self.max_target_sequence_length
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.decoder_targets,
                                                                      logits=self.logits)
            target_weights = tf.sequence_mask(self.decoder_targets_length, max_times,
                                              dtype=self.logits.dtype)
            # target_weights target_weights is a zero-one matrix of the same size as decoder_outputs.
            # QUESTION: 这一步到底需不需要
            self.train_loss = (tf.reduce_sum(crossent * target_weights) / self.batch_size)
            optim = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
            self.train_op = optim.minimize(self.train_loss)


    def init_op(self):
        self.init_op = tf.global_variables_initializer()

    def train(self, cn_train_data, en_train_data):
        
        saver = tf.train.Saver(tf.global_variables())
        with tf.Session() as sess:
            sess.run(self.init_op)

            for epoch in range(self.epoch_num):
                self.run_one_epoch(sess,cn_train_data, en_train_data, epoch, saver)

 
    def run_one_epoch(self, sess, cn_train_data, en_train_data, epoch, saver):
        """
        """

        num_batches = (len(cn_train_data) + self.batch_size - 1) // self.batch_size
        
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # batch的时候进行id操作
        train_batches = batch_yield(cn_train_data, en_train_data, self.batch_size, self.cn_word2id_dict, self.en_word2id_dict)

        for step, (encoder_inputs, decoder_inputs) in enumerate(train_batches):
            # encoder_inputs, decoder_inputs 是已进行过id化处理的

            sys.stdout.write(' processing: {} batch / {} batches.'.format(step + 1, num_batches) + '\r')
            step_num = epoch * num_batches + step + 1
            
            # 进行pad操作
            # 得到id化句子padding后的结果，以及每个句子真实长度
            encoder_seq_list, encoder_len_list,decoder_seq_list, decoder_len_list = pad_sequences(encoder_inputs, 
                                                                                                  decoder_inputs, 
                                                                                                  self.cn_word2id_dict, 
                                                                                                  self.en_word2id_dict,
                                                                                                  pad_mark= '<PAD>')
            # print ("encoder_seq_list: ", len(encoder_seq_list), "encoder_len_list: ", len(encoder_len_list),
            #         "decoder_seq_list: ", len(decoder_seq_list), "decoder_len_list: ", len(decoder_len_list))

            feed_dict = { self.encoder_inputs: encoder_seq_list,
                          self.encoder_inputs_length: encoder_len_list,
                          self.decoder_targets: decoder_seq_list,
                          self.decoder_targets_length: decoder_len_list,
                         }

            _, loss_train = sess.run([self.train_op, self.train_loss], feed_dict=feed_dict)


            if step + 1 == 1 or (step + 1) % 300 == 0 or step + 1 == num_batches:

                print ('{} epoch {}, step {}, loss: {:.4}, global_step: {}'.format(start_time, epoch + 1, 
                                                                                   step + 1,loss_train, step_num))

            if step + 1 == num_batches:
                saver.save(sess, self.model_path, global_step = step_num)

        # QUESTION： 要不要每跑完一个epoch的全部batches就评价一次???

    def test(self, cn_sent, en_sent):
        """
        test的时候直接跑rnn_output.sample_id即可，但是需要先跑训练吗？ 如果先跑训练再跑test, args.mode怎么改？ decoder里面的不同操作怎么改？ 
        要在train训练好的基础上跑还是重新以args.mode=infer初始化模型然后decoder? 
        """
        all_candidates, all_references = [], []
        eos_id = self.en_word2id_dict['<EOS>']
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, self.model_path)

            batches = batch_yield(cn_sent, en_sent, self.batch_size, self.cn_word2id_dict, self.en_word2id_dict)

            for src,tgt in batches:

                encoder_seq_list, encoder_len_list,decoder_seq_list, decoder_len_list = pad_sequences(src,tgt, 
                                                                                              self.cn_word2id_dict, 
                                                                                              self.en_word2id_dict,
                                                                                              pad_mark= '<PAD>')
                # print ("PAD id: ", self.en_word2id_dict['<PAD>'])
                # print("EOS id: ", self.en_word2id_dict['<EOS>'])
                # print ("deocder_length: ", decoder_len_list)
                # print ("decoder pad length: ", [len(i) for i in decoder_seq_list])
                feed_dict = { self.encoder_inputs: encoder_seq_list,
                              self.encoder_inputs_length: encoder_len_list,
                              self.decoder_targets_length: decoder_len_list
                             }
            
                predict = sess.run([self.predicted_sample_id], feed_dict=feed_dict)
                predict = np.array(predict)
                # predict: [1,batch_size, self.max_target_sequence_length]  还需要通过真实长度截取吗？
                # bream时, predict : [1,batch_size, self.max_target_sequence_length, beam_width]
                # print (predict.shape)
                # predict[array([],[], ..., [])]
                # 在predict[0]中获取到当前Batch内所有句子的结果
                temp = predict[0]
                for item in temp:
                    if self.beam_width:
                        sent = []
                        for beam_item in item:
                            # li = [1,2,1,4,1,4]  Counter(li).most_common(1) [(1, 3)]
                            # 找到当前宽度中出现次数最多的预测词
                            most_common_res = Counter(list(beam_item)).most_common(1)[0][0]
                            # 将most_common_res加入到当前句子里
                            sent.append(most_common_res)
                        if eos_id in sent:
                            sent = sent[:sent.index(eos_id)]
                        sent_word = sentence2word(sent, self.en_id2word_dict)
                        # print("sent: ", sent_word)
                        all_candidates.append(sent_word)

                    else:
                        sent = list(item)
                        if eos_id in sent:
                            sent = sent[:sent.index(eos_id)]
                        sent_word = sentence2word(sent, self.en_id2word_dict)
                        all_candidates.append(sent_word)
        calculate_bleu(all_candidates, en_sent)
            # for sents in predict:
            #     # print (sents)
            # for sent in predict[0]:
            #     # print (sent)
            #     en_sent = [self.en_id2word_dict[en_id] for en_id in sent]
            #     print (en_sent)
            #     # for en_id in sent:
            #     #     print (en_id, self.en_id2word_dict[en_id])







