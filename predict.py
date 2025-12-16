import torch
from torch import nn
import jieba
import os
import re
from collections import OrderedDict
from tqdm import tqdm
import numpy as np

from config import *
from utils import try_all_gpus
from dataset import get_vocab, clear_data, tokenize
from models.BiRNN import BiRNN
from models.TextCNN import TextCNN

def predict_sentiment_batch(net, vocab, sentences, device):
    """Batch predict the sentiment of given sentences."""
    net.eval()
    
    # 文本清理
    cleaned_sentences = clear_data(sentences)
    
    # 对过短文本进行重复处理
    processed_sentences = []
    for sentence in cleaned_sentences:
        # 如果文本过短，进行重复
        if len(sentence.strip()) < 5:  # 长度阈值设为5个字符
            repeat_times = min(3, max(1, 10 // len(sentence.strip())))  # 重复次数：目标长度约10个字符
            sentence = sentence * repeat_times
        processed_sentences.append(sentence)
    
    # 分词
    all_tokens = [list(jieba.cut(sentence)) for sentence in processed_sentences]
    
    # 转换为词索引
    batch_indices = []
    valid_sentences = []
    short_text_flags = []  # 标记原始短文本
    
    unk_idx = vocab.unk
    pad_idx = vocab['<pad>']
    
    for i, tokens in enumerate(all_tokens):
        # 标记原始短文本（用于结果分析）
        original_length = len(cleaned_sentences[i].strip())
        is_short = original_length < 5
        short_text_flags.append(is_short)
        
        # 处理空文本
        if not tokens:
            tokens = ['<unk>']
        
        # 获取索引
        indices = [vocab.token_to_idx.get(token, unk_idx) for token in tokens]
        
        # 截断或填充
        if len(indices) < NUM_STEPS:
            indices.extend([pad_idx] * (NUM_STEPS - len(indices)))
        else:
            indices = indices[:NUM_STEPS]
        
        batch_indices.append(indices)
        valid_sentences.append(i)
    
    if not batch_indices:
        return ["无法识别的输入"] * len(sentences)
    
    # 批量预测
    batch_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device)
    
    with torch.no_grad():
        output = net(batch_tensor)
        probs = torch.softmax(output, dim=1)
        preds = torch.argmax(output, dim=1)
    
    # 构建结果
    results = []
    for i, (pred, prob, is_short) in enumerate(zip(preds, probs, short_text_flags)):
        confidence = prob.max().item()
        sentiment = '积极' if pred.item() == 1 else '消极'
        
        # 对短文本添加标记
        if is_short:
            results.append(f"{sentiment}(短文本处理)")
        elif confidence < 0.3:
            results.append(f"不确定({sentiment})")
        else:
            results.append(sentiment)
    
    return results

def load_model(net_name, vocab, devices):
    """Load the trained model."""
    if net_name == "BiRNN":
        embed_size, num_hiddens, num_layers = EMBED_SIZE, NUM_HIDDEN, NUM_LAYERS
        net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
    elif net_name == "TextCNN":
        embed_size, kernel_sizes, nums_channels = EMBED_SIZE, KERNEL_SIZES, NUMS_CHANNELS 
        net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
    else:
        raise ValueError(f"未知的模型名称: {net_name}")

    # 加载权重
    weight_path = os.path.join(WEIGHTS_DIR, f'{net_name}.pt')
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"模型权重文件未找到: {weight_path}")
    
    # 加载权重并处理DataParallel前缀
    state_dict = torch.load(weight_path, map_location='cpu')
    
    # 去掉module.前缀
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove 'module.'
        new_state_dict[name] = v
    
    net.load_state_dict(new_state_dict)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    return net

if __name__ == '__main__':
    # 加载词汇表
    vocab = get_vocab()
    
    # 配置
    devices = try_all_gpus()
    net_name = NET_NAME
    
    # 加载模型
    net = load_model(net_name, vocab, devices)
    print(f"成功加载模型 {net_name}")
    
    # 从test_data读取测试数据，并保存到test_predict_result.csv
    test_sentences = []
    
    test_file_path = os.path.join(TEST_DATA_DIR, 'test_weibo_comment.txt')
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"测试文件未找到: {test_file_path}")
        
    with open(test_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if len(line.strip()) >= MIN_FREQ:
                test_sentences.append(line.strip())

    print("\n\n"+"="*100+"\n"+"="*100+"\n\n")
    print(f"device: \t\t{devices}")
    print(f"batch_size: \t\t{BATCH_SIZE}")
    print(f"lr: \t\t\t{LR}")
    print(f"num_epochs: \t\t{NUM_EPOCHS}")
    print(net) 
    print("\n\n"+"="*100+"\n"+"="*100+"\n\n")
    
    print(f"开始预测 {len(test_sentences)} 条微博评论...")
    
    # 批量预测
    results = []
    batch_size = BATCH_SIZE  # 使用配置中的批次大小
    
    for i in tqdm(range(0, len(test_sentences), batch_size), desc='predict batches'):
        batch_sentences = test_sentences[i:i + batch_size]
        batch_results = predict_sentiment_batch(net, vocab, batch_sentences, devices[0])
        results.extend(batch_results)

    # 保存结果和句子
    result_file_path = os.path.join(TEST_DATA_DIR, f'{net_name}_predict_result.csv')
    with open(result_file_path, 'w', encoding='utf-8') as f:
        for sentence, result in zip(test_sentences, results):
            f.write(f"{result}，{sentence}\n")
    
    print(f"预测完成，结果已保存到 {result_file_path}")