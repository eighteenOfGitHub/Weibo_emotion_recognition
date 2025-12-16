import gradio as gr
import random
import time
from collections import Counter
import matplotlib.pyplot as plt
import io
import base64

# --- 模拟数据生成函数 (二级情感) ---
def generate_mock_data(query):
    """模拟根据搜索词爬取和分析微博评论 (二级情感: 积极/消极)"""
    # 模拟处理时间
    time.sleep(1) 
    
    # 简化为两级情感标签
    sentiments = ['积极 😊', '消极 😞']
    
    # 根据不同关键词，调整积极/消极的概率
    if "开心" in query or "高兴" in query or "胜利" in query:
        sentiment_weights = [0.8, 0.2] # 更倾向于积极
    elif "难过" in query or "生气" in query or "失败" in query:
        sentiment_weights = [0.2, 0.8] # 更倾向于消极
    else:
        sentiment_weights = [0.5, 0.5] # 默认平衡

    # 随机生成评论数量 (例如 5 到 15 条)
    num_comments = random.randint(5, 15)
    
    mock_comments = []
    sentiment_labels = []

    for _ in range(num_comments):
        # 随机选择一条评论模板
        templates = [
            f"关于'{query}'，我觉得{random.choice(['很棒', '不错', '很失望', '糟糕透了'])}！",
            f"刚看到有关{query}的消息，{random.choice(['心情愉悦', '感觉不太好', '五味杂陈'])}。",
            f"{query}？{random.choice(['强烈支持👍', '无法苟同👎', '吃瓜群众🍉'])}",
            f"对{query}的看法：{random.choice(['乐观📈', '悲观📉', '静待发展'])}"
        ]
        comment = random.choice(templates)
        
        # 根据权重随机分配情感标签 (现在只有两级)
        label = random.choices(sentiments, weights=sentiment_weights, k=1)[0]
        
        mock_comments.append((label, comment))
        sentiment_labels.append(label)
        
    return mock_comments, sentiment_labels


# --- 数据处理与可视化函数 (针对二级情感) ---
def analyze_and_visualize(comments_with_labels):
    """分析二级情感标签并生成饼图"""
    if not comments_with_labels:
        # 如果没有数据，返回空字符串和提示信息
        return "<div style='text-align: center; color: gray; font-size: 20px;'>暂无数据</div>"

    labels = [item[0] for item in comments_with_labels]
    count_dict = Counter(labels)
    
    # 固定顺序和颜色，适应二级分类
    all_labels = ['积极 😊', '消极 😞']
    sizes = [count_dict.get(label, 0) for label in all_labels]
    colors = ['#64b464', '#ee6e6e'] # 绿色 (积极), 红色 (消极)
    
    # 过滤掉计数为0的项
    filtered_labels = [label for label, size in zip(all_labels, sizes) if size > 0]
    filtered_sizes = [size for size in sizes if size > 0]
    filtered_colors = [color for color, size in zip(colors, sizes) if size > 0]

    # 创建饼图
    plt.figure(figsize=(6, 4))
    
    if filtered_sizes:
        wedges, texts, autotexts = plt.pie(
            filtered_sizes, 
            labels=filtered_labels, 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=filtered_colors,
            textprops={'fontsize': 12} # 调整标签字体大小
        )
        # 让百分比文字更清晰
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    else:
        # 所有数据都为0的情况
        plt.pie([1], labels=['无数据'], autopct='%1.1f%%', startangle=90, colors=['#cccccc'])
        
    plt.title('情感分析统计结果 (积极 vs. 消极)', fontsize=14)
    plt.axis('equal') # 保证画出的是圆形
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
    plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

    # 将图片保存到内存中的字节流
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight') # bbox_inches='tight' 防止标题被裁剪
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close() # 关闭图形，释放内存

    # 生成HTML格式的图片标签
    html_img = f'<div style="text-align: center;"><img src="data:image/png;base64,{image_base64}" alt="情感分析饼图" style="max-width: 100%; height: auto;" /></div>'
    
    return html_img


# --- 主要的 Gradio 接口函数 ---
def run_sentiment_analysis(query):
    """Gradio 调用的主要函数"""
    if not query.strip():
        # 如果输入为空，清空输出
        return "<div style='text-align: center; color: gray; font-size: 20px;'>请输入搜索词条</div>", []
        
    # 1. 获取模拟数据
    comments_with_labels, sentiment_labels = generate_mock_data(query)
    
    # 2. 分析并生成图表
    plot_html = analyze_and_visualize(comments_with_labels)
    
    # 3. 准备列表数据
    comment_list = [(label, comment) for label, comment in comments_with_labels]

    return plot_html, comment_list


# --- Gradio 界面构建 ---
with gr.Blocks(title="微博情感分析系统 (二级情感)") as demo:
    gr.Markdown("# 📊 微博情感分析系统 💬")
    # 更新说明文字
    gr.Markdown("🔍 输入您感兴趣的微博话题，查看大家的 **积极** 或 **消极** 情感倾向！")

    with gr.Row():
        with gr.Column(scale=3):
            input_query = gr.Textbox(
                label="📝 请输入想搜索的词条",
                placeholder="例如：人工智能, 新闻事件, ...",
                elem_id="input-box"
            )
            run_button = gr.Button("🚀 开始分析", variant="primary")
            
        with gr.Column(scale=2):
             # 添加一个有趣的图片或图标作为装饰
             gr.HTML("""
             <div style='display: flex; justify-content: center; align-items: center; height: 100%;'>
                 <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 24 24" fill="none" stroke="#3498db" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                     <path d="M23 3a10.9 10.9 0 0 1-3.14 1.53 4.48 4.48 0 0 0-7.86 3v1A10.66 10.66 0 0 1 3 4s-4 9 5 13a11.64 11.64 0 0 1-7 2c9 5 20 0 20-11.5a4.5 4.5 0 0 0-.08-.83A7.72 7.72 0 0 0 23 3z"></path>
                 </svg>
             </div>
             """)
    
    with gr.Row():
        # 统计结果区域
        with gr.Column():
            gr.Markdown("### 📈 统计结果")
            output_plot = gr.HTML(label="情感分布图") # 使用 HTML 组件来显示图片
        
        # 列表区域
        with gr.Column():
            gr.Markdown("### 📋 评论列表")
            # 使用 DataFrame 展示列表
            output_list = gr.DataFrame(
                label="",
                headers=["情感标签", "评论内容"],
                interactive=False,
                wrap=True,
                elem_id="comment-list"
            )

    # 设置按钮点击事件
    run_button.click(
        fn=run_sentiment_analysis,
        inputs=input_query,
        outputs=[output_plot, output_list]
    )
    
    # 设置回车键触发
    input_query.submit(
        fn=run_sentiment_analysis,
        inputs=input_query,
        outputs=[output_plot, output_list]
    )

# --- 自定义CSS美化界面 ---
demo.css = """
#input-box {
    font-size: 18px;
}
#comment-list table {
    font-size: 16px;
}
footer {
    visibility: hidden;
}
"""

# 启动应用
if __name__ == "__main__":
    demo.launch()