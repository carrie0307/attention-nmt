# 机器翻译模型

UCAS NLP homework

## RNN + Attention 的机器翻译模型

* 代码主要参考自： 

    * https://github.com/tensorflow/nmt/tree/tf-1.4

    * https://github.com/VectorFist/RNN-NMT/tree/master/NMT

    * https://github.com/princewen/tensorflow_practice/blob/master/nlp/chat_bot_seq2seq_attention/model.py


## Transformer

* 主要是代码学习，参考自

    * Transformer 代码实现  https://yuanxiaosc.github.io/2018/11/08/Transformer%E4%BB%A3%E7%A0%81%E5%AE%9E%E7%8E%B0/

    * Trnsformer 源码总结 https://blog.csdn.net/yiyele/article/details/81913031

    * 机器翻译模型 Transformer代码解析 https://blog.csdn.net/mijiaoxiaosan/article/details/74909076


## 运行命令

### Attention模块：

1. 进入 ./Attention/ 目录下: cd Attention

2. 以Greedy模式运行： python main.py --mode infer

3. 以BeamSearch模式运行： python main.py --mode infer --beam_width 5


### Transformer模块：

1. 进入 ./Transformer/ 目录下：  cd Transformer

2. python main.py --is_training 0   (运行会比Attention慢，运行过程会打印出bach信息，请稍安勿躁)


./Attention/目录下和./Transformer/目录下分别也写了对应的运行命令

### 所需依赖包：

tensorflow (1.8.0)

math

collections

nltk

warnings

numpy

random

pickle

os

argpars

time

sys

