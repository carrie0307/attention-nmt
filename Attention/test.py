# -*- coding: utf-8 -*-



with open('./data/en_dev.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        if len(line.split()) > 29:
            print ("----------------")

# import tensorflow as tf

# a = tf.ones([5, ], tf.int32) * 20

# ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
# decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word_to_idx['<go>']), ending], 1)

# w1 = tf.Variable(tf.random_normal([6, 3], stddev=1, seed=1))
# # ending = tf.strided_slice(w1, [0, 0], [6, -1], [1, 1])
# input_info = tf.concat([tf.fill([6, 1], 10.3), w1], 1)
# input_info2 = tf.concat([w1, tf.fill([6, 1], 10.3)], -1)
#
# with tf.Session() as sess :
#     # w1= tf.Variable([[1,2,3],[10,20,30]])
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     # print("e4")
#     print( sess.run(w1) )
#     # print ( sess.run(ending))
#     print (sess.run(input_info))
#     print (sess.run(input_info2))



