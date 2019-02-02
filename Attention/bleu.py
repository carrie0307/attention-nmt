# -*- coding: utf-8 -*-
import math
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
import warnings
warnings.filterwarnings('ignore')
# https://blog.csdn.net/guolindonggld/article/details/56966200   BLEU计算讲解

def get_ngram(sent, n):
    """
    获取一个句子的n-gram元素统计
    :param sent: 要划分的句子,eg.['this', 'is', 'small', 'test']
    :param n: n
    :return: eg.n=1 {'this ': 1, 'is ': 1, 'small ': 1, 'test ': 1}; n=2 {'this is ': 1, 'is small ': 1, 'small test ': 1}
    """

    length = len(sent)
    ngram_list = []
    for i, word in enumerate(sent):
        item = ''
        for j in range(0, n):
            if i + j < length:
                item = item + str(sent[i + j]) + " "
            else:
                j = j - 1
                break
        if j == n - 1:
            ngram_list.append(item)
    counter = dict(Counter(ngram_list))
    return counter


def calculate_p(c_counter, r_counter):
    """

    :param c_counter:
    :param r_counter:
    :return:
    """
    total = 0
    c_total = 0
    for word in c_counter:
        num = min(c_counter[word], r_counter.get(word,0))
        total += num
        c_total += c_counter[word]
    return round(total / c_total,2)

def ngram_one_sent(candidate, reference):
    """
    对一个句子的n=1,2,3,4 gram 进行计算
    :param candidate:
    :param reference:
    :return:
    """
    # 1-gram
    c_gram1 = get_ngram(candidate, 1)
    r_gram1 = get_ngram(reference, 1)
    p1 = calculate_p(c_gram1, r_gram1)
    p1 = p1 if p1 != 0 else 1
    # print (c_gram1, r_gram1)

    # 2-gram
    c_gram2 = get_ngram(candidate, 2)
    r_gram2 = get_ngram(reference, 2)
    p2 = calculate_p(c_gram2, r_gram2)
    p2 = p2 if p2 != 0 else 1

    # 3-gram
    c_gram3 = get_ngram(candidate, 3)
    r_gram3 = get_ngram(reference, 3)
    p3 = calculate_p(c_gram3, r_gram3)
    p3 = p3 if p3 != 0 else 1

    # 4-gram
    c_gram4 = get_ngram(candidate, 4)
    r_gram4 = get_ngram(reference, 4)
    p4 = calculate_p(c_gram4, r_gram4)
    p4 = p4 if p4 != 0 else 1

    return p1, p2, p3, p4


def brevity_penalty(candidate, reference):
    """
    计算乘法因子
    :param c: c 是候选译文中单词的个数
    :param r: r 是答案译文中与c最接近的译文单词个数
    :return:
    """

    c, r = len(candidate), len(reference)
    if c > r:
        penalty = 1
    else:
        penalty = math.exp(1 - (float(r) / c))
    return penalty

def calculate_bleu(candidates, references):
    """
    计算p
    :param candidates:
    :param references:
    :return:
    """
    w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25
    length = len(candidates)
    my_min_bleu, my_max_bleu, my_total = 10000, 0, 0
    nltk_min_bleu, nltk_max_bleu, nltk_total = 10000, 0, 0
    for candidate,reference in zip(candidates,references):

        # 获得一个句子的四个Pn
        p1, p2, p3, p4 = ngram_one_sent(candidate, reference)
        # print ("p1,p2,p3,p4: ", p1, p2, p3, p4)
        P = w1 * math.log(p1) + w2 * math.log(p2) + w3 * math.log(p3) + w4 * math.log(p4)
        penalty = brevity_penalty(candidate, reference)
        # print ("penalty: ", penalty)
        # 得到一个句子的bleu
        my_bleu = penalty * math.exp(P)
        # 统计Bleu的最大值、最小值和均值(先记录综合)
        my_total += my_bleu
        my_min_bleu = my_bleu if my_bleu <= my_min_bleu else my_min_bleu
        my_max_bleu = my_bleu if my_bleu >= my_max_bleu else my_max_bleu

        # nltk提供接口计算结果
        nltk_bleu = sentence_bleu([reference], candidate, weights=(1, 0, 0, 0))
        nltk_total += nltk_bleu
        nltk_min_bleu = nltk_bleu if nltk_bleu <= nltk_min_bleu else nltk_min_bleu
        nltk_max_bleu = nltk_bleu if nltk_bleu >= nltk_max_bleu else nltk_max_bleu

    my_bleu_avg = my_total / length
    print ("\n最终结果是: \n")

    print ("自己写的bleu计算结果：bleu_max = {}, bleu_min = {}, bleu_avg = {}\n".format(str(my_max_bleu),
                                                                                     str(my_min_bleu),
                                                                                     str(my_bleu_avg)))
    nltk_bleu_avg = nltk_total / length
    print("nltk的sentence_bleu计算结果：nltk_bleu_max = {}, nltk_bleu_min = {}, nltk_bleu_avg = {}".format(str(nltk_max_bleu),
                                                                                                            str(nltk_min_bleu),
                                                                                                            str(nltk_bleu_avg)))



if __name__ == '__main__':
    # candidate = [['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']]
    # reference = [['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']]
    candidate = [['the', 'united', 'states', 'of', 'the', 'united', 'states', 'of', 'the', 'united', 'states', 'of', 'the', 'united',
     'states', 'of', 'the', 'united', 'states', 'of', 'the', 'united', 'states', '.']]
    reference = [['one', 'was', 'called', 'the', '"', 'maintenance', 'of', 'regional', 'security', '"', 'or', '"', 'maintenance',
     'of', 'humanitarianism', ',', '"', 'and', 'consisted', 'of', 'military', 'attacks', 'on', 'these', 'nations', '.',
     '<EOS>']]
    p1, p2, p3, p4 = ngram_one_sent(candidate[0], reference[0])
    print (p1, p2, p3, p4)
    P = 0.25 * math.log(p1) + 0.25 * math.log(p2) + 0.25 * math.log(p3) + 0.25 * math.log(p4)
    print ("P: ", P)
    penalty = brevity_penalty(candidate, reference)
    print ("penalty: ", penalty)
    # 得到一个句子的bleu
    curr_sent_bleu = penalty * math.exp(P)
    print (curr_sent_bleu)
    # calculate_bleu(candidate, reference)