import os
import requests
import time
import random
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import concurrent.futures
from urllib.parse import urlparse

# ==============================================================================
# ### 用户配置区 ###
# ==============================================================================

# 1. 待处理的 PDF 文件 URL 列表
# 请将您需要转换的 PDF 的公开访问链接填入此列表。
# 示例:
# URLS_TO_PROCESS = [
#     "https://cdn-mineru.openxlab.org.cn/demo/example.pdf",
#     "https://arxiv.org/pdf/2307.09288.pdf"
# ]
URLS_TO_PROCESS = [
    # 在这里填入您的URL
    "https://cdn-mineru.openxlab.org.cn/demo/example.pdf",
]

# 2. 存放解析后 Markdown 文件的文件夹路径
OUTPUT_FOLDER = "transferred_papers_from_url"

# 3. 您的 API Token
# 请务必替换为您自己的有效 Token
API_TOKEN = 'Bearer eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI2MjQwMzE5MCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc1Mjg0Mzc4NCwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiMzhkZDYzMjAtNzQ3Ny00ZjhjLTgwNTYtMWE0NjliNWUyZDc4IiwiZW1haWwiOiIiLCJleHAiOjE3NTQwNTMzODR9.2MBgX_jgoq6Z6XZLcMoi7YLZuJdQ_Yb2GXRh8SA0KglP0LWQZjUOTsv-xpgrIjCNGf9nBrsyRtg9CmUjIt6g0Q'

# 4. API 基础 URL 和轮询间隔
BASE_URL = 'https://mineru.net/api/v4'
POLLING_INTERVAL = 10  # 每 10 秒查询一次状态

# 5. 并行下载的最大线程数
MAX_DOWNLOAD_WORKERS = 8

# 6. 【新增】任务提交参数
# 您可以在这里统一配置提交任务时的参数，例如语言、是否OCR等。
# 更多参数请参考 API 文档：https://mineru.net/docs/api/extract
TASK_PARAMETERS = {
    "enable_formula": True,   # 是否开启公式识别
    "enable_table": True,     # 是否开启表格识别
    "language": "auto",       # 'ch' 'en' 或 'auto' 自动识别
    "model_version": "v2"     # 'v1' 或 'v2'
    # "is_ocr": True,         # 如果需要对扫描件PDF强制OCR，请取消此行注释
}

# ==============================================================================
# ### 脚本主逻辑区 (通常无需修改) ###
# ==============================================================================

def get_filename_from_url(url):
    """从URL中提取文件名"""
    try:
        path = urlparse(url).path
        filename = os.path.basename(path)
        if not filename:
            # 如果路径为空或以'/'结尾, 尝试使用一个备用名称
            return f"file_{random.randint(1000, 9999)}.pdf"
        return filename
    except Exception:
        return f"file_{random.randint(1000, 9999)}.pdf"


def download_and_save_single_file(task):
    """
    处理单个文件的函数：下载、解压、保存。
    这个函数将被多个线程并行调用。
    """
    if task.get("state") != "done":
        return None  # 如果任务不是完成状态，则直接跳过

    # 在URL模式下，API返回的 file_name 可能是从URL解析的
    file_name = task.get("file_name")
    zip_url = task.get("full_zip_url")

    if not file_name:
        file_name = f"unknown_file_{random.randint(1000,9999)}.zip"
        print(f"  ⚠️ 任务结果中未找到文件名，使用随机名称: {file_name}")

    if not zip_url:
        return f"  ❌ 文件 '{file_name}' 已完成但未提供下载链接。"
    
    try:
        # 下载zip文件到内存
        zip_response = requests.get(zip_url, timeout=180)  # 设置超时时间
        zip_response.raise_for_status()
        
        # 在内存中解压
        with ZipFile(BytesIO(zip_response.content)) as z:
            md_file_path_in_zip = None
            for item in z.namelist():
                if item.lower().endswith('.md'):
                    md_file_path_in_zip = item
                    break
            
            if md_file_path_in_zip:
                md_content = z.read(md_file_path_in_zip).decode('utf-8')
                base_name = os.path.splitext(file_name)[0]
                output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.md")
                
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(md_content)
                return f"  ✅ 成功保存 '{file_name}' 的结果到: {output_path}"
            else:
                return f"  ❌ 在 '{file_name}' 的结果压缩包中未找到 .md 文件。"

    except Exception as e:
        return f"  ❌ 下载或处理 '{file_name}' 时出错: {str(e)}"


def process_urls():
    """主处理函数 - URL版本"""
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    api_headers = {'Content-Type': 'application/json', 'Authorization': API_TOKEN}
    
    # 限制单次处理数量，防止超出API限制（文档说200）
    url_count = min(len(URLS_TO_PROCESS), 150) 
    urls = random.sample(URLS_TO_PROCESS, url_count) if len(URLS_TO_PROCESS) > url_count else URLS_TO_PROCESS

    if not urls:
        print("`URLS_TO_PROCESS` 列表中没有找到任何URL。")
        return
    print(f"找到 {len(urls)} 个URL待处理。")
    
    # --- 步骤 1: 直接提交URL解析任务 ---
    print("\n--- 步骤 1: 正在提交URL解析任务... ---")
    
    # 根据API文档构建 `files` 字段
    files_payload = [{"url": u} for u in urls]
    
    # 将通用参数与文件列表合并
    data_payload = {**TASK_PARAMETERS, "files": files_payload}

    try:
        response = requests.post(f'{BASE_URL}/extract/task/batch', headers=api_headers, json=data_payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if result.get("code") != 0:
            print(f"提交URL解析任务失败: {result.get('msg')}")
            return
            
        batch_id = result.get("data", {}).get("batch_id")
        if not batch_id:
            print("提交任务成功，但未能从响应中获取 batch_id。")
            print("完整响应:", result)
            return
            
        print(f"✅ 成功提交任务，批次ID: {batch_id}")

        # --- 步骤 2: 轮询处理状态 (原步骤3) ---
        print("\n--- 步骤 2: 任务已提交，开始轮询处理状态... ---")
        query_url = f"{BASE_URL}/extract-results/batch/{batch_id}"
        final_results = None
        while True:
            try:
                response = requests.get(query_url, headers=api_headers, timeout=30)
                if response.status_code != 200:
                    print(f"\n查询失败，状态码: {response.status_code}。稍后重试...")
                    time.sleep(POLLING_INTERVAL)
                    continue
                
                result = response.json()
                if result.get("code") != 0:
                    print(f"\n查询API返回错误: {result.get('msg')}。稍后重试...")
                    time.sleep(POLLING_INTERVAL)
                    continue

                extract_results = result.get("data", {}).get("extract_result", [])
                if not extract_results:
                    print("\r  正在等待API创建任务...", end="")
                    time.sleep(POLLING_INTERVAL)
                    continue
                
                done_count = sum(1 for task in extract_results if task.get("state") == "done")
                failed_count = sum(1 for task in extract_results if task.get("state") == "failed")
                total_tasks = len(extract_results)
                
                print(f"\r  当前进度: {done_count} 个完成, {failed_count} 个失败, {total_tasks - done_count - failed_count} 个进行中...", end="")
                
                if (done_count + failed_count) >= total_tasks:
                    print("\n所有任务处理完毕！")
                    final_results = extract_results
                    break
            except requests.exceptions.RequestException as e:
                print(f"\n轮询时网络错误: {e}。稍后重试...")

            time.sleep(POLLING_INTERVAL)

        if not final_results:
            print("未能获取到最终结果，程序终止。")
            return
            
        # --- 步骤 3: 并行下载并保存结果 (原步骤4) ---
        print("\n--- 步骤 3: 正在并行下载并保存结果... ---")
        
        tasks_to_process = [task for task in final_results if task.get("state") == "done"]
        if not tasks_to_process:
            print("没有成功完成的任务可供下载。")
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
                future_to_task = {executor.submit(download_and_save_single_file, task): task for task in tasks_to_process}
                
                for future in concurrent.futures.as_completed(future_to_task):
                    result_message = future.result()
                    if result_message:
                        print(result_message)

        print("\n所有文件处理完成！")
        
    except requests.exceptions.RequestException as e:
        print(f"网络请求出错: {str(e)}")
    except Exception as e:
        print(f"处理过程中发生未知错误: {str(e)}")


if __name__ == "__main__":
    if '请' in API_TOKEN or len(API_TOKEN) < 20:
        print("错误：请先在脚本中配置您的有效 API_TOKEN。")
    else:
        process_urls()

