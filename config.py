import torch
from torch import nn
from d2l import torch as d2l
import os

from models.BiRNN import BiRNN
from models.TextCNN import TextCNN
from utils import try_all_gpus, TokenEmbedding

# data sourec

DATA_DIR = '.\data'
SOURCE_DATA_PATH = os.path.join(DATA_DIR, 'weibo_senti_100k.csv')
SOURCE_DATA_INFO_PATH = os.path.join(DATA_DIR, 'source_dataset_infos.json')
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'train.csv')
TRAIN_DATA_INFO_PATH = os.path.join(DATA_DIR, 'train_infos.json')

# dataset

NUM_STEPS = 128
MIN_FREQ = 5
BATCH_SIZE = 64

"""
1 长度覆盖

cut_coverage 里 128 步已覆盖 96.93 % 的样本，再升到 256 只能提升 3 个百分点，却要多一倍的显存/时间，性价比低；
64 步又丢掉近 25 % 的信息，精度损失大。128 是“拐点”。

2 平均/最大长度

text_average_len=44，mode=12，最大 216。
128 ≈ 3×平均长度，既能容纳大部分长句，又把极长尾截断，减少噪声。

3 类别与规模

12 万样本、2 分类，不算大；min_freq 设 5 可把出现 ≤4 次的极低频词（往往是错别字、乱码）当成 <unk>，词表大小能压缩到 1/3 左右，
后续 embedding 层参数量下降，训练更快，又不会损失主要信息。
设 10 会过度削词表，可能把有效情感词也滤掉；设 2 则词表过大，embedding 矩阵大半是长尾噪声。

4 经验验证

在同类中文短评/微博任务里，128+min_freq=5 几乎总是最先试的“黄金组合”，能快速得到不错 baseline，再调也省时间。
"""

# train

DEVICES = try_all_gpus()
NET_NAME = "TextCNN" # "TextCNN" "BiRNN"


EMBED_SIZE, NUM_HIDDEN, NUM_LAYERS = 300, 300, 4
KERNEL_SIZES, NUMS_CHANNELS = [3, 4, 5], [100, 100, 100]

LR, NUM_EPOCHS = 0.001, 5

WEIGHTS_DIR = ".\weights"
LOG_DIR = '.\log'

if __name__ == '__main__':
    pass

# weibo_spider
HEADERS_PATH = 'req_header.json'
TEST_DATA_DIR = '.\\test_data'

