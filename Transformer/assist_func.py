# -*- coding: utf-8 -*-

"""
    Graph所用的一些辅助函数：
    multihead_attention  Transformer中的多头Attention操作
    feedback_forward     TransFormer中的前馈网络
    normalize            Transformer中的norm操作

    embedding            根据lookup_table查询词向量
    label_smoothed       标签平滑操作
    posotion_embedding   sin位置编码
"""
from __future__ import print_function
import tensorflow as tf
import numpy as np


def multihead_attention(queries, keys, num_units=None, num_heads=8, dropout_rate=0, is_training=True,
                        causality=False, scope="multihead_attention", reuse=None):
    """
    多头attention计算(本次实现里key和values，因此只写了keys)
    https://www.imooc.com/article/51468
    :param queries:
    :param keys:
    :param num_units: attention的大小
    :param num_heads: head的个数
    :param dropout_rate: dropout比率
    :param is_training: 是否训练标识
    :param causality: 是否进行屏蔽，如果为True,则进行attention时未来的Units都被屏蔽
    :param scope:
    :param reuse:
    :return:  attention操作后的张量，shape = (N, T_q, C)
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        # 先对Q,K，V进行全连接的变换
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        # 将上一步变换的Q,K,V从最后一维分成num_heads=8份，并把这些分开的张量在第一个维度拼接起来得到Q_,K_,V_进行后续运算
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        # Q * K转置: 这一步将K转置后和Q_进行了矩阵乘法的操作，也就是在通过点成方法进行attention score的计算
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        # 除以调节因子
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        # 这里的目的是，让key值的unit为0的key对应的attention score极小，这一甲醛计算value时相当于对结果不产生影响
        # 首先用reduce_sum将最后一个维度上的值加起来
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        # Causality 标识是否屏蔽未来序列的信息(解码器self attention的时候不能看到自己之后的哪些信息)
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        # 对Attention score进行softmax操作
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        # 要被屏蔽的，时本身不懈怠信息或暂时不利用其信息的内容
        # query mask要将初始值为0的queries屏蔽
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        # 前三步与Key Mask方法类似，这里用softmax后的outputs的值与query_masks相乘，因此这一步后需要mask的全职会乘0，不需要mask
        # 的乘以之前取的正数的sign为1所以全职不变，从而实现了query_mask目的
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        # 用outputs和V_加权和计算出多个头attention的结果
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        # 上步得到的是多头attention的结果在第一个维度叠着，所以把它们split开重新concat到最后一个维度上
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        # outputs加上一开始的queries, 是残差的操作
        outputs += queries

        # Normalize
        outputs = normalize(outputs)  # (N, T_q, C)

    return outputs


def feedforward(inputs, num_units=[2048, 512], scope="multihead_attention", reuse=None):
    """
    将多头attention的输出送入全连接的前馈网络
    :param inputs: shape为[N,T,C]的张量
    :param num_units: 隐藏节点个数
    :param scope:
    :param reuse:
    :return:
    """

    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs,
                  "filters": num_units[0],
                  "kernel_size": 1,
                  "activation": tf.nn.relu,
                  "use_bias": True}
        # 利用一维卷积进行网络的设计
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs,
                  "filters": num_units[1],
                  "kernel_size": 1,
                  "activation": None,
                  "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        # 加上inputs的残差
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs


def normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
    """
    Transformer中的normalize操作
    :param inputs: 输入
    :param epsilon: 一个很小的数值，防止计算中一些错误的出现
    :param scope:  定义tf中的variable_scope
    :param reuse:
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def embedding(lookup_table, inputs, num_units, scale = False, scope = "embedding", reuse = None):
    """
    查询子词向量
    :param lookup_table:
    :param inputs:
    :param num_units:
    :param scale:
    :param scope:
    :param reuse:
    :return: 词向量表示的输入
    """
    outputs = tf.nn.embedding_lookup(lookup_table, inputs)

    # 根据num_units对**进行调整
    if scale:
        outputs = outputs * (num_units ** 0.5)

    return outputs


def label_smoothing(inputs, epsilon = 0.1):
    """
    对Label进行平滑操作
    把之前one_hot中的0改成了一个很小的数，1改成了一个较接近于1的数
    :param inputs: labels
    :param epsilon: 平滑率
    :return:
    """

    K = inputs.get_shape().as_list()[-1]
    return ((1-epsilon) * inputs) + (epsilon / K)

def positional_encoding(inputs, num_units, zero_pad=True, scale=True, scope="positional_encoding", reuse=None):

    """
    位置编码
    :param inputs:
    :param num_units:
    :param zero_pad:
    :param scale:
    :param scope:
    :param reuse:
    :return:
    """

    N, T = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units**0.5

        return outputs