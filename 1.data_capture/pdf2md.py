import os
import requests
import time
import random
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import concurrent.futures

# ==============================================================================
# ### 用户配置区 ###
# ==============================================================================

# 1. 存放待处理 PDF 文件的文件夹路径
INPUT_FOLDER = "origin_papers/accept-oral"

# 2. 存放解析后 Markdown 文件的文件夹路径
OUTPUT_FOLDER = "transferred_papers"

# 3. 您的 API Token
API_TOKEN = 'Bearer eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI2MjQwMzE5MCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc1Mjg0Mzc4NCwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiMzhkZDYzMjAtNzQ3Ny00ZjhjLTgwNTYtMWE0NjliNWUyZDc4IiwiZW1haWwiOiIiLCJleHAiOjE3NTQwNTMzODR9.2MBgX_jgoq6Z6XZLcMoi7YLZuJdQ_Yb2GXRh8SA0KglP0LWQZjUOTsv-xpgrIjCNGf9nBrsyRtg9CmUjIt6g0Q'

# 4. API 基础 URL 和轮询间隔
BASE_URL = 'https://mineru.net/api/v4'
POLLING_INTERVAL = 10  # 每 10 秒查询一次状态

# 5. 并行下载和处理的最大线程数
# 这个值决定了同时有多少个文件在被下载和处理。
# 建议范围 5-10，可以根据您的网络带宽调整。
MAX_DOWNLOAD_WORKERS = 8 

# ==============================================================================
# ### 脚本主逻辑区 ###
# ==============================================================================

def download_and_save_single_file(task):
    """
    【新增】处理单个文件的函数：下载、解压、保存。
    这个函数将被多个线程并行调用。
    """
    if task.get("state") != "done":
        return None # 如果任务不是完成状态，则直接跳过

    file_name = task.get("file_name")
    zip_url = task.get("full_zip_url")

    if not zip_url:
        return f"  ❌ 文件 '{file_name}' 已完成但未提供下载链接。"
    
    try:
        # 下载zip文件到内存
        zip_response = requests.get(zip_url, timeout=180) # 设置超时时间，防止单个任务卡死
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
                return f"  ✅ 成功保存 '{file_name}' 到: {output_path}"
            else:
                return f"  ❌ 在 '{file_name}' 的结果压缩包中未找到 .md 文件。"

    except Exception as e:
        return f"  ❌ 下载或处理 '{file_name}' 时出错: {str(e)}"


def process_pdf_files():
    """主处理函数"""
    Path(INPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    api_headers = {'Content-Type': 'application/json', 'Authorization': API_TOKEN}
    pdf_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.pdf')]
    paper_count = min(len(pdf_files), 20)
    pdf_files = random.sample(pdf_files, paper_count)
    if not pdf_files:
        print(f"在 '{INPUT_FOLDER}' 文件夹中没有找到PDF文件。")
        return
    print(f"找到 {len(pdf_files)} 个PDF文件待处理: {pdf_files}")
    
    print("\n--- 步骤 1: 正在请求上传URL... ---")
    data_payload = {"files": [{"name": f} for f in pdf_files]}
    
    try:
        response = requests.post(f'{BASE_URL}/file-urls/batch', headers=api_headers, json=data_payload)
        response.raise_for_status()
        result = response.json()
        if result.get("code") != 0:
            print(f"获取上传URL失败: {result.get('msg')}")
            return
        batch_id = result["data"]["batch_id"]
        file_url_map = dict(zip(pdf_files, result["data"]["file_urls"]))
        print(f"成功获取到上传URL，批次ID: {batch_id}")

        print("\n--- 步骤 2: 正在上传PDF文件... ---")
        for filename, upload_url in file_url_map.items():
            pdf_path = os.path.join(INPUT_FOLDER, filename)
            try:
                with open(pdf_path, 'rb') as f:
                    print(f"  正在上传 {filename}...")
                    res_upload = requests.put(upload_url, data=f)
                    if res_upload.status_code == 200:
                        print(f"  ✅ 成功上传: {filename}")
                    else:
                        print(f"  ❌ 上传失败: {filename}, 状态码: {res_upload.status_code}, 响应: {res_upload.text}")
            except Exception as e:
                print(f"  ❌ 上传文件 {filename} 时出错: {str(e)}")

        print("\n--- 步骤 3: 文件上传完成，开始轮询处理状态... ---")
        query_url = f"{BASE_URL}/extract-results/batch/{batch_id}"
        final_results = None
        while True:
            response = requests.get(query_url, headers=api_headers)
            if response.status_code != 200: time.sleep(POLLING_INTERVAL); continue
            result = response.json()
            if result.get("code") != 0: time.sleep(POLLING_INTERVAL); continue
            extract_results = result.get("data", {}).get("extract_result", [])
            if not extract_results: time.sleep(POLLING_INTERVAL); continue
            done_count = sum(1 for task in extract_results if task.get("state") == "done")
            failed_count = sum(1 for task in extract_results if task.get("state") == "failed")
            total_tasks = len(extract_results)
            print(f"\r  当前进度: {done_count} 个完成, {failed_count} 个失败, {total_tasks - done_count - failed_count} 个进行中...", end="")
            if (done_count + failed_count) == total_tasks:
                print("\n所有任务处理完毕！")
                final_results = extract_results
                break
            time.sleep(POLLING_INTERVAL)

        if not final_results:
            print("未能获取到最终结果，程序终止。")
            return
            
        print("\n--- 步骤 4: 正在并行下载并保存结果... ---")
        
        # 筛选出所有已完成的任务
        tasks_to_process = [task for task in final_results if task.get("state") == "done"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
            # 使用 executor.map 将 download_and_save_single_file 函数应用到每个任务上
            # 它会自动处理线程分配和结果收集
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
        process_pdf_files()
