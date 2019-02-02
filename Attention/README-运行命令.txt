以greedy进行解码的运行命令：

python main.py --mode infer


以beamsearch进行解码的运行命令：

python main.py --mode infer --beam_width 5


以上两条命令将分别自动加载已保存的模型运行