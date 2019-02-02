# encoding=utf-8
from gensim.models import word2vec
import data_process
import numpy as np
import pickle


data = data_process.read_corpus('source.txt')
sentences = [x[0] for x in data]
print ("constructing model ...")
model = word2vec.Word2Vec(sentences, min_count=1,size=120)
print ("read wordid...")
words_id,_ = data_process.load_wordid("word2id.pkl")
embedding = []
print ("construct vectors...")
# 根据word_id中词的顺序构建起词向量
embedding = [model[word] for word in words_id]
embedding = np.array(embedding)
print (embedding.shape)
# 将词向量写入pickle文件，便于以后直接加载
with open('data/word2vector_source_embedding120.pkl', 'wb') as fw:
    pickle.dump(embedding, fw)
# words_embedding,_ = data_process.load_wordid("word2vector_source_embedding100.pkl")
# print (words_embedding.shape)