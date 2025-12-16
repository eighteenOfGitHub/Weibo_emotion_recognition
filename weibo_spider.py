from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager   # pip install webdriver-manager
from time import sleep
from selenium.webdriver.common.by import By
import pandas as pd
import os
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm
import time

from config import TEST_DATA_DIR

options = webdriver.ChromeOptions()
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option('useAutomationExtension', False)
driver = webdriver.Chrome(service= Service(r"D:\Chrome\chromedriver-win64\chromedriver.exe"),
                          options=options)
driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
    "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
})
wait = WebDriverWait(driver, 600)  # 最长等你 10 分钟扫码

def crawl_weibo_search_comment(keyword):

    # 登录

    driver.get(f"https://s.weibo.com/weibo?q={keyword}")
    # 下面滚动、解析、保存 csv 与 Edge 完全一致
    print('>>> 请用手机微博扫码登录（超时 10 分钟）...')
    # ---------- 3. 阻塞等待：URL 一旦变成目标搜索页就说明登录成功 ----------
    wait.until(lambda d: d.current_url.startswith('https://s.weibo.com/weibo?q='))
    print('>>> 扫码成功，已进入搜索页！')
    # 模拟下拉滚动多次
    for i in tqdm(range(3), desc=">>> 正在滚动加载更多微博..."):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        sleep(2)

    data = []
    # 尝试点击评论按钮 - 修复点击被拦截问题
    try:
        # 使用JavaScript点击避免被遮挡
        comment_buttons = WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "i.woo-font--comment"))
        )
        
        if comment_buttons:
            print(f">>> 找到 {len(comment_buttons)} 个评论按钮")
            # 点击所有评论按钮
            for i, button in enumerate(comment_buttons[:-1]):
                try:
                    driver.execute_script("arguments[0].click();", button)
                    print(f">>> 已点击第 {i+1} 个评论按钮")
                    time.sleep(3)  # 等待评论加载
                except Exception as e:
                    print(f"❌ 点击第 {i+1} 个评论按钮失败:", e)
                    continue
            
            # 等待所有评论内容加载
            time.sleep(3)
            
            # 提取所有评论内容
            try:
                # 等待评论加载完成
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".card-review"))
                )

                # 获取所有评论块
                comments = driver.find_elements(By.CSS_SELECTOR, ".card-review")
                print(f">>> 找到 {len(comments)} 条评论")
                
                for i, comment in enumerate(tqdm(comments, desc=">>> 正在提取评论...")):
                    try:
                        # 精准定位评论文本
                        text_elem = comment.find_element(By.CSS_SELECTOR, ".txt")
                        full_text = text_elem.text.strip()

                        # 提取冒号后的评论内容
                        if ":" in full_text:
                            comment_text = full_text.split(":", 1)[1].strip()
                        else:
                            comment_text = full_text

                        print(f">>> 提取到评论{i+1}：", comment_text)
                        data.append(comment_text)

                    except Exception as e:
                        print("❌ 提取失败:", e)
            except Exception as e:
                print("❌ 提取评论失败:", str(e))
        else:
            print("❌ 未找到评论按钮")
            
    except Exception as e:
        print("❌ 点击评论失败:", str(e))
        # 尝试其他方式获取微博内容
        print(">>> 尝试直接提取微博内容...")
        try:
            weibos = driver.find_elements(By.CSS_SELECTOR, "[action-type='feed_list_item']")
            for weibo in weibos:
                try:
                    content = weibo.find_element(By.CSS_SELECTOR, ".wbpro-feedpage-content").text
                    if content:
                        data.append(content)
                except:
                    continue
        except Exception as e:
            print("❌ 提取微博内容失败:", str(e))

    except Exception as e:
        print("❌ 爬取过程出错:", str(e))
    
    finally:
        driver.quit()
    
    return data


if __name__ == '__main__':
    data = crawl_weibo_search_comment("高考")
    print(f">>> 共提取 {len(data)} 条数据")
    
    # 保存为文本txt
    with open(os.path.join(TEST_DATA_DIR, 'test_weibo_comment.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(data))
    
    print(">>> 数据已保存到 test_weibo_comment.txt")