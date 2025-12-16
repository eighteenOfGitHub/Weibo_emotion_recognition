# download_weights.py
import os
import zipfile
import requests
from tqdm import tqdm

def download_file(url, dest):
    """下载文件，带进度条"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    with open(dest, 'wb') as f, tqdm(
        desc=dest,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)

def extract_zip(zip_path, extract_to):
    """解压 zip 文件"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

if __name__ == "__main__":
    # 配置
    URL = "https://github.com/eighteenOfGitHub/Weibo_emotion_recognition/releases/download/v1.0/weights.zip"
    ZIP_PATH = "weights.zip"
    EXTRACT_TO = "weights"

    # 下载
    if not os.path.exists(ZIP_PATH):
        print("正在下载模型权重...")
        download_file(URL, ZIP_PATH)
    else:
        print("权重压缩包已存在，跳过下载。")

    # 解压
    if not os.path.exists(EXTRACT_TO):
        print("正在解压权重...")
        extract_zip(ZIP_PATH, EXTRACT_TO)
        print(f"✅ 权重已解压到 {EXTRACT_TO}/")
    else:
        print(f"📁 {EXTRACT_TO}/ 已存在，跳过解压。")