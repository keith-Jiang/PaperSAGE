import os
import requests
import time
import random
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import concurrent.futures
import logging
import sys

# --- 0. 日志配置 ---
# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- 1. 全局配置区 ---
now = datetime.now()
formatted_date = now.strftime("%Y%m%d")
# 存放待处理 PDF 文件的文件夹路径
INPUT_FOLDER = f"origin_papers/{formatted_date}"
# INPUT_FOLDER = "origin_papers/accept-oral"
# 存放解析后 Markdown 文件的文件夹路径
OUTPUT_FOLDER = f"transferred_papers/{formatted_date}"
# 您的 API Token
API_TOKEN = 'Bearer eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI2MjQwMzE5MCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc1Mjg0Mzc4NCwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiMzhkZDYzMjAtNzQ3Ny00ZjhjLTgwNTYtMWE0NjliNWUyZDc4IiwiZW1haWwiOiIiLCJleHAiOjE3NTQwNTMzODR9.2MBgX_jgoq6Z6XZLcMoi7YLZuJdQ_Yb2GXRh8SA0KglP0LWQZjUOTsv-xpgrIjCNGf9nBrsyRtg9CmUjIt6g0Q'
# API 基础 URL 和轮询间隔
BASE_URL = 'https://mineru.net/api/v4'
POLLING_INTERVAL = 10  # 每 10 秒查询一次状态
# 并行下载和处理的最大线程数
MAX_DOWNLOAD_WORKERS = 4


def download_and_save_single_file(task):
    """
    【并行工作函数】处理单个文件的函数：下载、解压、保存。
    返回一个描述结果的字符串。
    """
    if task.get("state") != "done":
        return None  # 如果任务不是完成状态，则直接跳过

    file_name = task.get("file_name")
    zip_url = task.get("full_zip_url")

    if not zip_url:
        return f"❌ 文件 '{file_name}' 已完成但未提供下载链接。"
    
    try:
        # 下载zip文件到内存
        zip_response = requests.get(zip_url, timeout=180)
        zip_response.raise_for_status()
        
        # 在内存中解压
        with ZipFile(BytesIO(zip_response.content)) as z:
            md_file_path_in_zip = next((item for item in z.namelist() if item.lower().endswith('.md')), None)
            
            if md_file_path_in_zip:
                md_content = z.read(md_file_path_in_zip).decode('utf-8')
                base_name = os.path.splitext(file_name)[0]
                output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}.md")
                
                with open(output_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(md_content)
                return f"✅ 成功保存 '{file_name}' 到: {output_path}"
            else:
                return f"❌ 在 '{file_name}' 的结果压缩包中未找到 .md 文件。"

    except Exception as e:
        return f"❌ 下载或处理 '{file_name}' 时出错: {str(e)}"


def process_pdf_files():
    """主处理函数"""
    Path(INPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    api_headers = {'Content-Type': 'application/json', 'Authorization': API_TOKEN}
    pdf_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.pdf')]
    paper_count = min(len(pdf_files), 20)
    if paper_count < len(pdf_files):
        pdf_files = random.sample(pdf_files, paper_count)
    
    if not pdf_files:
        logger.info(f"在 '{INPUT_FOLDER}' 文件夹中没有找到PDF文件。")
        return
    logger.info(f"找到并准备处理 {len(pdf_files)} 个PDF文件。")
    
    logger.info("--- 步骤 1: 正在请求上传URL... ---")
    data_payload = {"files": [{"name": f} for f in pdf_files]}
    
    try:
        response = requests.post(f'{BASE_URL}/file-urls/batch', headers=api_headers, json=data_payload)
        response.raise_for_status()
        result = response.json()
        if result.get("code") != 0:
            logger.error(f"获取上传URL失败: {result.get('msg')}")
            return
        batch_id = result["data"]["batch_id"]
        file_url_map = dict(zip(pdf_files, result["data"]["file_urls"]))
        logger.info(f"成功获取到上传URL，批次ID: {batch_id}")

        logger.info("--- 步骤 2: 正在上传PDF文件... ---")
        for filename, upload_url in file_url_map.items():
            pdf_path = os.path.join(INPUT_FOLDER, filename)
            try:
                with open(pdf_path, 'rb') as f:
                    logger.info(f"  正在上传 {filename}...")
                    res_upload = requests.put(upload_url, data=f)
                    if res_upload.status_code == 200:
                        logger.info(f"  ✅ 成功上传: {filename}")
                    else:
                        logger.error(f"  ❌ 上传失败: {filename}, 状态码: {res_upload.status_code}, 响应: {res_upload.text}")
            except Exception as e:
                logger.error(f"  ❌ 上传文件 {filename} 时出错: {str(e)}")

        logger.info("--- 步骤 3: 文件上传完成，开始轮询处理状态... ---")
        query_url = f"{BASE_URL}/extract-results/batch/{batch_id}"
        final_results = None
        while True:
            response = requests.get(query_url, headers=api_headers)
            if response.status_code != 200: 
                time.sleep(POLLING_INTERVAL)
                continue
            result = response.json()
            if result.get("code") != 0: 
                time.sleep(POLLING_INTERVAL)
                continue
            extract_results = result.get("data", {}).get("extract_result", [])
            if not extract_results: 
                time.sleep(POLLING_INTERVAL)
                continue
                
            done_count = sum(1 for task in extract_results if task.get("state") == "done")
            failed_count = sum(1 for task in extract_results if task.get("state") == "failed")
            total_tasks = len(extract_results)
            # 注意：logging不支持\r来覆盖行，所以每次都会打印新的一行日志，这是标准行为
            logger.info(f"当前进度: {done_count} 个完成, {failed_count} 个失败, {total_tasks - done_count - failed_count} 个进行中...")
            
            if (done_count + failed_count) == total_tasks:
                logger.info("所有任务处理完毕！")
                final_results = extract_results
                break
            time.sleep(POLLING_INTERVAL)

        if not final_results:
            logger.error("未能获取到最终结果，程序终止。")
            return
            
        logger.info(f"--- 步骤 4: 正在使用 {MAX_DOWNLOAD_WORKERS} 个线程并行下载并保存结果... ---")
        tasks_to_process = [task for task in final_results if task.get("state") == "done"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
            future_to_task = {executor.submit(download_and_save_single_file, task): task for task in tasks_to_process}
            
            for future in concurrent.futures.as_completed(future_to_task):
                result_message = future.result()
                if result_message:
                    # 根据消息内容判断是成功还是失败，并使用不同日志级别
                    if "✅" in result_message:
                        logger.info(result_message)
                    else:
                        logger.error(result_message)

        logger.info("所有文件处理完成！")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"网络请求出错: {str(e)}")
    except Exception as e:
        logger.error(f"处理过程中发生未知错误: {str(e)}", exc_info=True) # exc_info=True 会记录完整的堆栈跟踪

if __name__ == "__main__":
    if '请' in API_TOKEN or len(API_TOKEN) < 20:
        logger.error("错误：请先在脚本中配置您的有效 API_TOKEN。")
    else:
        process_pdf_files()