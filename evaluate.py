import torch
from torch import nn
import os
from collections import OrderedDict
import datetime

from dataset import load_data_db
from config import *
from utils import try_all_gpus, evaluate_accuracy_gpu
from train import get_net

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
                if y[i] == 1 and pred[i] == 1:
                    tp += 1
                elif y[i] == 0 and pred[i] == 1:
                    fp += 1
                elif y[i] == 0 and pred[i] == 0:
                    tn += 1
                elif y[i] == 1 and pred[i] == 0:
                    fn += 1
    
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
    
    # 错误分析
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
        for i, (pred, true) in enumerate(zip(predictions, true_labels)):
            if pred != true and len(errors) < 10:  # 显示前10个错误
                if i < len(test_texts):
                    errors.append({
                        'index': i,
                        'text': test_texts[i],
                        'true_label': '积极' if true == 1 else '消极',
                        'pred_label': '积极' if pred == 1 else '消极'
                    })
        
        error_count = sum(1 for p, t in zip(predictions, true_labels) if p != t)
        print(f"总共 {len(predictions)} 个样本，错误 {error_count} 个")
        print(f"错误率: {error_count / len(predictions):.4f}")
        print(f"准确率: {1 - error_count / len(predictions):.4f}")
        
        if errors:
            print(f"\n前 {len(errors)} 个错误样本:")
            for i, error in enumerate(errors, 1):
                print(f"{i:2d}. 文本: {error['text'][:50]}{'...' if len(error['text']) > 50 else ''}")
                print(f"    真实标签: {error['true_label']}, 预测标签: {error['pred_label']}\n")
        else:
            print("没有发现错误样本!")
            
    except Exception as e:
        print(f"错误分析失败: {e}")

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

if __name__ == '__main__':
    try:
        test_acc = evaluate_test_set()
        print(f"\n测试集评估完成!")
        print(f"最终准确率: {test_acc:.4f}")
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()