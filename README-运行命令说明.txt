Attention模块：

1. 进入 ./Attention/ 目录下: cd Attention

2. 以Greedy模式运行： python main.py --mode infer

3. 以BeamSearch模式运行： python main.py --mode infer --beam_width 5


Transformer模块：

1. 进入 ./Transformer/ 目录下：  cd Transformer

2. python main.py --is_training 0   (运行会比Attention慢，运行过程会打印出bach信息，请稍安勿躁)


./Attention/目录下和./Transformer/目录下分别也写了对应的运行命令

所需依赖包：
tensorflow (1.8.0)
math
collections
nltk
warnings
numpy
random
pickle
os
argparse
time
sys