def test_spider_cookie():
    import requests, re
    uid = '2348668214'          # 换成你的 UID
    cookie = ('SCF=AlGf9rHLTiEJM1OTw392XCw1JpSp69B2ryG1D7ktWWQOiY4WkX0Gb9OAu6Nygi0jtpEi8lSPVo5rz-NBdwbAcq0.; SUB=_2A25EOgW7DeRhGeFI61cU8y_FwzWIHXVnNgdzrDV6PUJbktAbLRWnkW1NfX4LmQ-a6T0ToZEkqt_mQ9x7O_Asr7sV; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WF.OJWViLzN5YzQSWB20SIY5NHD95QNSo5fSKep1Kn4Ws4DqcjMi--NiK.Xi-2Ri--ciKnRi-zNS0q7SK-0eK.R1Btt; SSOLoginState=1765701099; ALF=1768293099'
    )  
    html = requests.get(f'https://weibo.cn/u/{uid}', # 网址是手机的，是 .cn ， 不是 .com
                        headers={'Cookie': cookie, 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36'},
                        timeout=10).text
    print('[DEBUG]', html)
    try:
        print('昵称', re.findall(r'昵称[：:](.+?)<br/>', html)[0])
        print('微博数', re.findall(r'微博\[(\d+)\]', html)[0])
    except IndexError:
        print('Cookie 无效或 UID 错误')

import torch
from torch import nn
import os
from collections import OrderedDict
import datetime
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jieba
from collections import Counter

from dataset import load_data_db
from config import *
from utils import try_all_gpus, evaluate_accuracy_gpu
from train import get_net

def save_evaluation_results(net_name, accuracy, precision, recall, f1):
    """保存评估结果到文件"""
    # 确保日志目录存在
    os.makedirs(LOG_DIR, exist_ok=True)
    
    result_file = os.path.join(LOG_DIR, f'{net_name}_evaluation.txt')
    
    # 获取当前时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f"模型: {net_name}\n")
        f.write(f"评估时间: {current_time}\n")
        f.write("="*50 + "\n")
        f.write(f"准确率 (Accuracy):  {accuracy:.4f}\n")
        f.write(f"精确率 (Precision): {precision:.4f}\n")
        f.write(f"召回率 (Recall):    {recall:.4f}\n")
        f.write(f"F1分数:            {f1:.4f}\n")
        f.write("="*50 + "\n")
    
    print(f"评估结果已保存到: {result_file}")

def evaluate_test_set():
    """评估测试集性能"""
    print("开始评估测试集性能...")
    
    # 加载数据
    batch_size = BATCH_SIZE
    _, test_iter, vocab = load_data_db(batch_size)
    devices = try_all_gpus()
    net_name = NET_NAME
    
    # 创建模型
    net = get_net(net_name, vocab)
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    
    # 加载权重
    weight_path = os.path.join(WEIGHTS_DIR, f'{net_name}.pt')
    best_weight_path = os.path.join(WEIGHTS_DIR, f'{net_name}_best.pt')
    
    # 优先加载最佳模型
    if os.path.exists(best_weight_path):
        weight_path = best_weight_path
        print(f"加载最佳模型权重: {best_weight_path}")
    elif os.path.exists(weight_path):
        print(f"加载模型权重: {weight_path}")
    else:
        raise FileNotFoundError(f"模型权重文件未找到: {weight_path}")
    
    # 加载权重
    try:
        state_dict = torch.load(weight_path, map_location='cpu')
        
        # 处理DataParallel前缀
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        net.module.load_state_dict(new_state_dict)
        print(f"成功加载模型权重")
    except Exception as e:
        print(f"加载模型权重失败: {e}")
        raise
    
    # 评估
    test_acc = evaluate_accuracy_gpu(net, test_iter, devices[0])
    print(f"测试集准确率: {test_acc:.4f}")
    
    # 详细分析
    accuracy, precision, recall, f1 = analyze_test_performance(net, test_iter, devices[0], vocab)
    
    # 保存评估结果
    save_evaluation_results(net_name, accuracy, precision, recall, f1)
    
    return test_acc

def analyze_test_performance(net, test_iter, device, vocab):
    """详细分析测试集性能"""
    if isinstance(net, nn.Module):
        net.eval()
    
    correct, total = 0, 0
    tp, fp, tn, fn = 0, 0, 0, 0  # 混淆矩阵统计
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            pred = torch.argmax(y_hat, dim=1)
            
            # 保存预测结果用于分析
            predictions.extend(pred.cpu().numpy())
            true_labels.extend(y.cpu().numpy())
            
            # 统计基本准确率
            correct += (pred == y).sum().item()
            total += y.size(0)
            
            # 混淆矩阵统计
            for i in range(len(y)):
                true_label = y[i].item()
                pred_label = pred[i].item()
                
                if true_label == 1 and pred_label == 1:
                    tp += 1  # 真正例
                elif true_label == 0 and pred_label == 1:
                    fp += 1  # 假正例
                elif true_label == 0 and pred_label == 0:
                    tn += 1  # 真负例
                elif true_label == 1 and pred_label == 0:
                    fn += 1  # 假负例
    
    accuracy = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "="*60)
    print("测试集详细性能分析:")
    print("="*60)
    print(f"准确率 (Accuracy):  {accuracy:.4f}")
    print(f"精确率 (Precision): {precision:.4f}")
    print(f"召回率 (Recall):    {recall:.4f}")
    print(f"F1分数:            {f1:.4f}")
    
    print("\n混淆矩阵:")
    print(f"                预测积极    预测消极")
    print(f"实际积极          {tp:6d}      {fn:6d}")
    print(f"实际消极          {fp:6d}      {tn:6d}")
    
    # 错误分析和数据统计
    error_analysis(predictions, true_labels)
    
    return accuracy, precision, recall, f1

def error_analysis(predictions, true_labels):
    """错误分析"""
    print("\n" + "="*60)
    print("错误样本分析:")
    print("="*60)
    
    # 重新加载测试数据用于显示
    try:
        from dataset import read_db
        _, test_data = read_db()
        test_texts = test_data[0]
        test_labels = test_data[1]
        
        # 找出错误预测的样本
        errors = []
        error_details = []
        
        for i, (pred, true) in enumerate(zip(predictions, true_labels)):
            if pred != true and i < len(test_texts):
                error_type = ""
                if true == 1 and pred == 0:
                    error_type = "假负例(积极->消极)"
                elif true == 0 and pred == 1:
                    error_type = "假正例(消极->积极)"
                
                errors.append({
                    'index': i,
                    'text': test_texts[i],
                    'true_label': '积极' if true == 1 else '消极',
                    'pred_label': '积极' if pred == 1 else '消极',
                    'error_type': error_type
                })
                
                error_details.append({
                    'text': test_texts[i],
                    'type': "假负例" if true == 1 and pred == 0 else "假正例",
                    'true_label': '积极' if true == 1 else '消极',
                    'pred_label': '积极' if pred == 1 else '消极',
                    'length': len(test_texts[i])
                })
        
        error_count = sum(1 for p, t in zip(predictions, true_labels) if p != t)
        print(f"总共 {len(predictions)} 个样本，错误 {error_count} 个")
        print(f"错误率: {error_count / len(predictions):.4f}")
        print(f"准确率: {1 - error_count / len(predictions):.4f}")
        
        if errors:
            print(f"\n前 {min(10, len(errors))} 个错误样本:")
            for i, error in enumerate(errors[:10], 1):
                print(f"{i:2d}. 文本: {error['text'][:50]}{'...' if len(error['text']) > 50 else ''}")
                print(f"    真实标签: {error['true_label']}, 预测标签: {error['pred_label']}\n")
        else:
            print("没有发现错误样本!")
            
        # 保存错误数据
        save_wrong_data(error_details, test_texts, test_labels, predictions)
        
        # 统计分析
        statistical_analysis(error_details, test_texts, test_labels)
            
    except Exception as e:
        print(f"错误分析失败: {e}")
        import traceback
        traceback.print_exc()

def save_wrong_data(error_details, test_texts, test_labels, predictions):
    """保存错误数据到CSV文件"""
    # 确保目录存在
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    
    wrong_data_file = os.path.join(TEST_DATA_DIR, 'wrong_data.csv')
    
    # 准备数据
    wrong_data = []
    for i, (text, true_label, pred_label) in enumerate(zip(test_texts, test_labels, predictions)):
        if true_label != pred_label:
            error_type = "假负例" if true_label == 1 else "假正例"
            wrong_data.append([error_type, text])
    
    # 保存到CSV
    df = pd.DataFrame(wrong_data, columns=['错误类型', '评论内容'])
    df.to_csv(wrong_data_file, index=False, encoding='utf-8-sig')
    print(f"错误数据已保存到: {wrong_data_file}")

def statistical_analysis(error_details, test_texts, test_labels):
    """统计分析错误数据"""
    print("\n" + "="*60)
    print("统计分析:")
    print("="*60)
    
    # 确保日志目录存在
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # 1. 长度分析
    all_lengths = [len(text) for text in test_texts]
    error_lengths = [error['length'] for error in error_details]
    
    # 2. 错误类型统计
    error_types = [error['type'] for error in error_details]
    error_type_counter = Counter(error_types)
    
    # 3. 标签分布
    true_labels_dist = Counter(['积极' if label == 1 else '消极' for label in test_labels])
    error_true_labels = [error['true_label'] for error in error_details]
    error_true_labels_counter = Counter(error_true_labels)
    
    # 4. 分词分析
    all_tokens = []
    error_tokens = []
    
    for text in test_texts:
        tokens = list(jieba.cut(text))
        all_tokens.extend(tokens)
    
    for error in error_details:
        tokens = list(jieba.cut(error['text']))
        error_tokens.extend(tokens)
    
    # 5. 高频词分析
    all_tokens_counter = Counter(all_tokens)
    error_tokens_counter = Counter(error_tokens)
    
    # 构建分析结果
    analysis_result = {
        "时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "总体统计": {
            "总样本数": len(test_texts),
            "错误样本数": len(error_details),
            "错误率": len(error_details) / len(test_texts) if len(test_texts) > 0 else 0
        },
        "长度分析": {
            "所有样本平均长度": float(np.mean(all_lengths)) if all_lengths else 0,
            "所有样本长度标准差": float(np.std(all_lengths)) if all_lengths else 0,
            "错误样本平均长度": float(np.mean(error_lengths)) if error_lengths else 0,
            "错误样本长度标准差": float(np.std(error_lengths)) if error_lengths else 0,
            "所有样本长度分布": dict(Counter([f"{l//10*10}-{l//10*10+9}" for l in all_lengths])),
            "错误样本长度分布": dict(Counter([f"{l//10*10}-{l//10*10+9}" for l in error_lengths]))
        },
        "错误类型分析": dict(error_type_counter),
        "真实标签分布": dict(true_labels_dist),
        "错误样本真实标签分布": dict(error_true_labels_counter),
        "高频词分析": {
            "所有样本Top20高频词": dict(all_tokens_counter.most_common(20)),
            "错误样本Top20高频词": dict(error_tokens_counter.most_common(20))
        }
    }
    
    # 保存统计信息到JSON
    info_file = os.path.join(LOG_DIR, 'wrong_data_info.json')
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(analysis_result, f, ensure_ascii=False, indent=2)
    print(f"统计信息已保存到: {info_file}")
    
    # 生成图表和分析报告
    generate_charts_and_report(all_lengths, error_lengths, error_type_counter, 
                              error_true_labels_counter, all_tokens_counter, 
                              error_tokens_counter, analysis_result)

def generate_charts_and_report(all_lengths, error_lengths, error_type_counter, 
                              error_true_labels_counter, all_tokens_counter, 
                              error_tokens_counter, analysis_result):
    """生成图表和分析报告"""
    
    # 设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        pass
    
    # 1. 长度分布直方图
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.hist(all_lengths, bins=50, alpha=0.7, label='所有样本', color='blue', edgecolor='black', linewidth=0.5)
    plt.hist(error_lengths, bins=50, alpha=0.7, label='错误样本', color='red', edgecolor='black', linewidth=0.5)
    plt.xlabel('文本长度')
    plt.ylabel('频数')
    plt.title('文本长度分布对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. 错误类型饼图
    plt.subplot(2, 3, 2)
    if error_type_counter:
        error_types = list(error_type_counter.keys())
        counts = list(error_type_counter.values())
        plt.pie(counts, labels=error_types, autopct='%1.1f%%', startangle=90)
        plt.title('错误类型分布')
    else:
        plt.text(0.5, 0.5, '无错误数据', ha='center', va='center')
        plt.title('错误类型分布')
    
    # 3. 真实标签分布
    plt.subplot(2, 3, 3)
    labels = list(error_true_labels_counter.keys())
    counts = list(error_true_labels_counter.values())
    if labels:
        bars = plt.bar(labels, counts, color=['green', 'orange'])
        plt.xlabel('真实标签')
        plt.ylabel('错误数量')
        plt.title('错误样本真实标签分布')
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, '无错误数据', ha='center', va='center')
        plt.title('错误样本真实标签分布')
    
    # 4. 长度箱线图
    plt.subplot(2, 3, 4)
    if all_lengths and error_lengths:
        data = [all_lengths, error_lengths]
        plt.boxplot(data, labels=['所有样本', '错误样本'])
        plt.ylabel('文本长度')
        plt.title('文本长度箱线图')
    else:
        plt.text(0.5, 0.5, '数据不足', ha='center', va='center')
        plt.title('文本长度箱线图')
    
    # 5. 高频词对比 (Top 10)
    plt.subplot(2, 3, 5)
    if all_tokens_counter and error_tokens_counter:
        all_top10 = dict(all_tokens_counter.most_common(10))
        error_top10 = dict(error_tokens_counter.most_common(10))
        
        all_words = list(all_top10.keys())
        all_counts = list(all_top10.values())
        error_counts = [error_top10.get(word, 0) for word in all_words]
        
        x = np.arange(len(all_words))
        width = 0.35
        
        plt.bar(x - width/2, all_counts, width, label='所有样本', alpha=0.8)
        plt.bar(x + width/2, error_counts, width, label='错误样本', alpha=0.8)
        plt.xlabel('高频词')
        plt.ylabel('频数')
        plt.title('高频词对比 (Top 10)')
        plt.xticks(x, all_words, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, '无词汇数据', ha='center', va='center')
        plt.title('高频词对比')
    
    # 6. 长度分段统计
    plt.subplot(2, 3, 6)
    if all_lengths:
        # 按长度分段统计
        length_ranges = [0, 10, 20, 50, 100, 200, max(all_lengths) + 1]
        range_labels = ['0-10', '11-20', '21-50', '51-100', '101-200', f'201-{max(all_lengths)}']
        
        all_range_counts = []
        error_range_counts = []
        
        for i in range(len(length_ranges) - 1):
            start, end = length_ranges[i], length_ranges[i+1]
            all_count = sum(1 for l in all_lengths if start <= l < end)
            error_count = sum(1 for l in error_lengths if start <= l < end)
            all_range_counts.append(all_count)
            error_range_counts.append(error_count)
        
        x = np.arange(len(range_labels))
        width = 0.35
        
        plt.bar(x - width/2, all_range_counts, width, label='所有样本', alpha=0.8)
        plt.bar(x + width/2, error_range_counts, width, label='错误样本', alpha=0.8)
        plt.xlabel('长度范围')
        plt.ylabel('样本数量')
        plt.title('不同长度范围样本分布')
        plt.xticks(x, range_labels, rotation=45, ha='right')
        plt.legend()
    else:
        plt.text(0.5, 0.5, '无长度数据', ha='center', va='center')
        plt.title('长度分段统计')
    
    plt.tight_layout()
    chart_file = os.path.join(LOG_DIR, 'wrong_data_analysis_charts.png')
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"分析图表已保存到 {chart_file}")
    

if __name__ == '__main__':
    try:
        test_acc = evaluate_test_set()
        print(f"\n测试集评估完成!")
        print(f"最终准确率: {test_acc:.4f}")
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()