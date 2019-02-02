# -*- coding: utf-8 -*-
# import tensorflow as tf
# import numpy as np


# t1 = [[1, 2, 3], [4, 5, 6]]
# t2 = [[1],[1]]
#
# t3 = tf.concat([t1, t2], 1)
# t4 = tf.concat([t1, t2], -1)
# t5 = tf.fill([2, 1], 1)
# t6 = tf.ones_like(t1[:, :1]) * 2
#
# with tf.Session() as sess :
#     # w1= tf.Variable([[1,2,3],[10,20,30]])
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     # print("e4")
#     print( sess.run(t5) )
#     print( sess.run(t6) )


from nltk.translate.bleu_score import corpus_bleu,sentence_bleu

# references = [['this', 'is', 'a', 'test']]
# candidates = [['this', 'is', 'a', 'test']]
# score = corpus_bleu(references, candidates)
# score2 = sentence_bleu(references, candidates[0], weights=(0.25, 0.25, 0.25, 0.25))
# print(score)
# print (score2)

c = ['since', 'of', 'president', 'from', 'on', 'on', 'on', 'on', 'on', 'to', 'to', 'on', 'to', 'on', 'on', 'to', ',', 'to', 'to', 'to', ',', 'to', 'to', ',', 'to', 'to', 'to', 'to', 'to']
r = [['however', ',', 'since', 'former', 'president', 'bush', 'approved', 'selling', 'f-16', 'fighters', 'to', 'taiwan', 'in', 'september', '1992', ',', 'the', '"', '17', 'august', 'communique', '"', 'has', 'existed', 'in', 'name', 'only', '.', '<EOS>']]
# c = ['however', ',', 'since', 'former', 'president', 'bush', 'approved', 'selling', 'f-16', 'fighters', 'to', 'taiwan', 'in', 'september', '1992', ',', 'the', '"', '17', 'august', 'communique', '"', 'has', 'existed', 'in', 'name', 'only', '.', '<EOS>']
score2 = sentence_bleu(r, c, weights=(0.25, 0, 0, 0))
print (score2)