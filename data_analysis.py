import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import numpy as np
from pprint import pprint

from config import SOURCE_DATA_PATH, SOURCE_DATA_INFO_PATH,TRAIN_DATA_INFO_PATH, TRAIN_DATA_PATH, DATA_DIR

"""
1 只做了数学上的统计，以及后期模型需要的统计
2 在数据清洗上的统计没有做统计，做出来为了后期数据清洗
    a 网文叠词，重叠符号
    b 符号表情：^_^
    c 常用停用词
    d 网络人名，地名，机构名
"""
    
def load_train_data_info(path=SOURCE_DATA_INFO_PATH):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    
def load_train_data(path=SOURCE_DATA_PATH):
    return pd.read_csv(path, encoding='utf-8')
    

def analyze_train_data(data: pd.DataFrame):
    train_data_info = load_train_data_info()
    if 'statistics' not in train_data_info:
        train_data_info['statistics'] = {}

    lens = data['sentence'].fillna('').astype(str).str.len()
    total = lens.size

    # 统计总数量
    train_data_info['statistics']['total'] = len(data)
    # 统计lebel的类型
    train_data_info['statistics']['label_type_count'] = len(data['label'].unique())
    # 统计正面数量
    train_data_info['statistics']['positive'] = len(data[data['label'] == 1])
    # 统计负面数量
    train_data_info['statistics']['negative'] = len(data[data['label'] == 0])
    # 统计文本平均长度
    train_data_info['statistics']['text_average_len'] = round(lens.mean())
    # 统计文本中最长文本长度
    train_data_info['statistics']['text_max_len'] = int(lens.max())
    # 统计文本中最短文本长度
    train_data_info['statistics']['text_min_len'] = (int(lens.min()))
    # 统计文本中长度众数列表
    train_data_info['statistics']['text_mode_len'] = list(map(int, lens.mode()))
    # 统计文本长度频数
    freq = lens.value_counts().sort_index()

    x_length = list(map(int, freq.index))
    y_count = list(map(int, freq.values))
    train_data_info['statistics']['text_len_freq'] = {
        "x_length": x_length,  # 长度值
        "y_count": y_count,  # 频数
        'length': len(freq)
    }

    # 3. 截断长度 & 覆盖率（常用 4 个档位）
    cut_points = [64, 128, 256]
    cum_count = np.cumsum(y_count)   # 累计样本量

    coverage = {}
    for cut in cut_points:
        # 找到最后一个 <= cut 的索引
        idx = np.searchsorted(x_length, cut, side='right') - 1
        cover_rate = cum_count[idx] / total * 100
        coverage[str(cut)] = round(cover_rate, 2)
    train_data_info['statistics']['cut_coverage'] = coverage

    return train_data_info

def save(file_path, data):
    if isinstance(data, dict):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def plot_length_frequency(x_list, y_list, save_path=None, step=10):
    plt.figure(figsize=(8, 4))
    plt.xticks(np.arange(0, len(x_list)+1, step))   # 颗粒度 = 10
    plt.xticks(rotation=45, ha='right')
    plt.plot(x_list, y_list, linewidth=2)
    plt.xlabel('Sentence Length')   # 单独写
    plt.ylabel('Frequency')
    plt.title('Length Frequency Distribution')  

    plt.savefig(save_path)
    plt.show()

def analyze_source_data_info():
    train_data = load_train_data()
    train_data_info = analyze_train_data(train_data)
    print(train_data_info)
    save(SOURCE_DATA_INFO_PATH, train_data_info)
    print(train_data_info['statistics']['text_len_freq']['x_length'])
    print(train_data_info['statistics']['text_len_freq']['y_count'])

    # train_data_info = load_train_data_info()
    # x_list = train_data_info['statistics']['text_len_freq']['x_length']
    # y_list = train_data_info['statistics']['text_len_freq']['y_count']
    # save_freq_path = os.path.join(DATA_DIRNAME, 'text_len_freq.png')
    # plot_length_frequency(x_list, y_list, save_freq_path, step=10)

def analyze_train_data_info():
    # train_data = load_train_data(TRAIN_DATA_PATH)
    # train_data_info = analyze_train_data(train_data)
    # print(train_data_info)
    # save(TRAIN_DATA_INFO_PATH, train_data_info)
    # print(train_data_info['statistics']['text_len_freq']['x_length'])
    # print(train_data_info['statistics']['text_len_freq']['y_count'])

    train_data_info = load_train_data_info(TRAIN_DATA_INFO_PATH)
    x_list = train_data_info['statistics']['text_len_freq']['x_length']
    y_list = train_data_info['statistics']['text_len_freq']['y_count']
    save_freq_path = os.path.join(DATA_DIR, 'train_text_len_freq.png')
    plot_length_frequency(x_list, y_list, save_freq_path, step=10)


def main():
    # analyze_source_data_info()
    analyze_train_data_info()


if __name__ == '__main__':
    main()