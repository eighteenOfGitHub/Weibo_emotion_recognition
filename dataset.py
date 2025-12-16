import os
import torch
import pandas as pd 
import random
import jieba
import re
from tqdm import tqdm

from vocab import Vocab
from config import  TRAIN_DATA_PATH, SOURCE_DATA_PATH, \
                    NUM_STEPS, MIN_FREQ, BATCH_SIZE

def clear_data(data):

    del_words = ['//', r'\]', r'\[','http\S+', r'#.*?#']     # 要删的词
    del_pat = re.compile('|'.join(del_words))       # 提前编译
    at_to_colon_1 = re.compile(r'@[^:]*:')          # 去用户名 1
    at_to_colon_2 = re.compile(r'@[^:]*：')         # 去用户名 2
    dup_sym_pat = re.compile(r'([^\w\s])\1{1,}')    # 去重复符号
    # 去重复情感叠词
    double_word = ['哈哈', '亲亲', '爱你', '吐', '泪', '抓狂', '呵', '害羞', '思考']
    double_pat = [re.compile(r'%s{2,}' % word) for word in double_word]

    out = []
    for sent in tqdm(data, desc='clear data'):
        tmp = re.sub(r'\s+', '', sent)  # 去除空格
        tmp = del_pat.sub('', tmp)      # 逐个替换
        tmp = at_to_colon_1.sub('', tmp)   # 删 @ 到 :
        tmp = at_to_colon_2.sub('', tmp)   # 删 @ 到 ：
        tmp = dup_sym_pat.sub(r'\1', tmp) # 去重符号
        for i in range(len(double_pat)):
            tmp = double_pat[i].sub(double_word[i], tmp) # 去重叠词
        out.append(tmp)

    return out

def clear_source_data():
    # 读取csv文件
    data = pd.read_csv(SOURCE_DATA_PATH)
    result = clear_data(data['sentence'])
    data['sentence'] = result
    # 去空值
    NaN_count = 0
    for i in tqdm(range(len(data)), desc='clear NaN'):
        if data['sentence'][i] == '':
            data = data.drop(i)
            NaN_count += 1

    print('NaN count:', NaN_count)

    data.to_csv(TRAIN_DATA_PATH, index=False)  


def read_db():
    train_data = pd.read_csv(TRAIN_DATA_PATH)
    labels = train_data['label'].to_list()
    text = train_data['sentence'].to_list()

    # 打乱数据集，80%为训练集，20%为测试集
    
    len_train = int(len(text) * 0.8) 
    random.seed(1)
    ids = [i for i in range(len(text))]
    random.shuffle(ids)

    train_text = [text[i] for i in ids[:len_train]]
    test_text = [text[i] for i in ids[len_train:]]
    train_label = [labels[i] for i in ids[:len_train]]
    test_label = [labels[i] for i in ids[len_train:]]
    return (train_text, train_label), (test_text, test_label)

def tokenize(lines):
    tokenize_lines = [[i for i in jieba.cut(line)] for line in tqdm(lines, desc='tokenize lines')]
    return tokenize_lines

def truncate_pad(line, num_steps, padding_token): # 截长补短
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate 
    return line + [padding_token] * (num_steps - len(line))  # Pad

def load_array(data_arrays, batch_size, is_train=True):
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

def get_vocab():
    min_freq = MIN_FREQ
    train_data, _ = read_db()
    train_tokens = tokenize(train_data[0])
    vocab = Vocab(train_tokens, min_freq)
    return vocab


def load_data_db(batch_size):
    """返回数据迭代器和数据集的词表"""

    num_steps = NUM_STEPS
    min_freq = MIN_FREQ

    train_data, test_data = read_db()

    train_tokens = tokenize(train_data[0])
    test_tokens = tokenize(test_data[0])

    print("tokenize done...")

    vocab = Vocab(train_tokens, min_freq)

    print("vocab done...")

    train_features = torch.tensor([truncate_pad( # 截断或填充
        vocab[line], num_steps, vocab['<pad>']) for line in tqdm(train_tokens, desc='train features')])
    test_features = torch.tensor([truncate_pad(
        vocab[line], num_steps, vocab['<pad>']) for line in tqdm(test_tokens, desc='test features')])
    
    print("features done...")

    train_iter = load_array((train_features, torch.tensor(train_data[1])),
                                batch_size)
    test_iter = load_array((test_features, torch.tensor(test_data[1])),
                               batch_size,
                               is_train=False)
    
    print("load data done...")

    return train_iter, test_iter, vocab

def main():
    # clear_data()
    batch_size = BATCH_SIZE
    train_iter, test_iter, vocab = load_data_db(batch_size)
    for X, y in train_iter:
        print('X:', X.shape, ', y:', y.shape)
        break
    print('小批量数目：', len(train_iter))

def test():
    num_steps = NUM_STEPS
    min_freq = MIN_FREQ

    train_data, test_data = read_db()

    result = [i for i in jieba.cut(train_data[0][0])]

    # print(result)

    # clear_source_data()

    #test_re_compile()
def test_re_compile():
    double_word = ['哈哈', '亲亲', '爱你', '吐', '泪', '抓狂', '呵', '害羞', '思考']
    double_pat = [re.compile(r'%s{2,}' % word) for word in double_word]
    # 1. 看正则长啥样
    for pat in double_pat:
        print(pat.pattern)

    # 2. 拿条句子实测
    test = '哈哈哈哈，亲亲吻我，泪泪泪，思考思考思考'
    for word, pat in zip(double_word, double_pat):
        hit = pat.findall(test)
        if hit:
            print(f'{word} 命中 -> {hit}')

if __name__ == '__main__':
    main()
    # test()
