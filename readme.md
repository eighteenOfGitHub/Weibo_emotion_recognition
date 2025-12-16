
# 📊 微博情感分析系统（Weibo Emotion Recognition）
基于深度学习的中文微博评论情感分析工具，支持 积极/消极 二分类。
包含数据爬取、清洗、模型训练（BiRNN / TextCNN）、评估、预测及 Web 可视化界面。

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

🔧 功能特性
🕷️ 自动爬取：使用 Selenium 爬取微博搜索结果下的评论（需扫码登录）
🧹 智能清洗：去除 @用户、链接、表情符号、重复叠词等噪声
🧠 双模型支持：
BiRNN（双向 RNN）
TextCNN（多尺度卷积）
📈 全面评估：准确率、精确率、召回率、F1、混淆矩阵、错误分析
🌐 Web 应用：基于 Gradio 的交互式情感分析界面
📊 可视化报告：生成错误样本统计图表与高频词分析

📁 项目结构

.
├── weibo_spider.py # 微博评论爬虫（Selenium）
├── dataset.py # 数据加载、清洗、分词、构建词表
├── vocab.py # 词汇表管理
├── models/ # 模型定义（BiRNN, TextCNN）
│ ├── BiRNN.py
│ └── TextCNN.py
├── utils.py # 工具函数（GPU、嵌入、精度计算等）
├── config.py # 全局配置（路径、超参、设备等）
├── train.py # 模型训练入口
├── test.py # 测试集评估 + 错误分析 + 图表生成
├── evaluate.py # 快速评估脚本
├── predict.py # 批量预测微博评论情感
├── data_analysis.py # 数据集统计分析
├── app.py # Gradio Web 应用
├── data/ # 数据目录
│ ├── weibo_senti_100k.csv # 原始数据集（需自行下载）
│ └── train.csv # 清洗后数据
├── weights/ # 保存的模型权重
├── log/ # 训练日志、评估结果、分析图表
├── test_data/ # 测试数据（如爬取的评论）
├── requirements.txt # 依赖
└── README.md

⚙️ 环境依赖

bash

pip install -r requirements.txt

requirements.txt由pipreqs自动分析代码导入

💡 注意：若使用 GPU，请安装对应 CUDA 版本的 PyTorch。

🚀 快速开始
1. 准备数据
下载 [weibo_senti_100k](https://github.com/SophonPlus/ChineseNlpCorpus) 中文情感数据集
放入 ./data/weibo_senti_100k.csv
1. 配置参数（可选）
编辑 config.py 修改：
NET_NAME = "TextCNN" 或 "BiRNN"
BATCH_SIZE, LR, NUM_EPOCHS
路径、设备等
1. 训练模型
bash
python train.py
模型权重将保存至 ./weights/
1. 评估模型
bash
python test.py # 完整评估 + 错误分析 + 图表
或
python evaluate.py # 快速评估
1. 预测新评论
先爬取评论（或准备文本文件）：
bash
python weibo_spider.py # 搜索关键词“高考”，结果存为 test_weibo_comment.txt
运行预测：
bash
python predict.py
结果保存为 ./test_data/{NET_NAME}_predict_result.csv
1. 启动 Web 应用（模拟版）
bash
python app.py
打开浏览器访问 http://localhost:7860
⚠️ 当前 app.py 为模拟演示（未集成真实模型）。如需接入真实预测，请修改 app.py 调用 predict.py 中的函数。

📊 评估指标示例

运行 test.py 后，将生成：
log/{NET_NAME}_evaluation.txt：准确率、F1 等指标
log/wrong_data_info.json：错误样本统计
log/wrong_data_analysis_charts.png：长度分布、高频词对比等图表
test_data/wrong_data.csv：具体错误样本

📝 注意事项
微博爬虫需手动扫码登录，请确保网络畅通。
不要提交大文件（如模型权重、原始数据集）到 Git。建议在 .gitignore 中忽略：
gitignore
data/
weights/
.csv
.pt
__pycache__/
.vscode/
若遇 Git LFS 错误，请勿跟踪大文件，改用外部存储分享。

📜 License

本项目采用 [MIT License](LICENSE)。

🙏 致谢
数据集来源：[ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)
词向量：可选用 [THUCNews](http://thuctc.thunlp.org/) 或 [Word2Vec 中文预训练向量]
参考：《动手学深度学习》（D2L）

✅ 使用建议

1. 将上述内容保存为项目根目录下的 README.md
2. 在 GitHub 仓库页面刷新即可看到美观的介绍
3. 如需添加 截图（如 Gradio 界面、训练曲线、分析图表），可在 README.md 中插入：
markdown
![Training Curve](log/TextCNN.png)
