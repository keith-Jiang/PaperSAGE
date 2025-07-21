# =================================================================
#  修改后的完整代码
# =================================================================

import requests
import json
import re
from datetime import datetime
from bs4 import BeautifulSoup
from summary_test_api import summary_func # 假设此模块可用
from pathlib import Path
import os

def scrape_arxiv_paper(arxiv_url):
    """
    爬取指定的ArXiv论文页面，提取核心信息以及用户指定的额外信息。

    :param arxiv_url: 论文的ArXiv链接 (例如: 'https://arxiv.org/abs/1706.03762')
    :return: 包含论文信息的JSON字符串，如果失败则返回None
    """
    
    # --- 1. 准备工作 ---
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        arxiv_id = arxiv_url.split('/')[-1]
    except IndexError:
        print("错误：无法从URL中解析ArXiv ID。请检查URL格式。")
        return None

    # --- 2. 爬取ArXiv页面，获取基础信息作为备用 ---
    print(f"正在从ArXiv爬取基础信息: {arxiv_url}")
    try:
        response = requests.get(arxiv_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
    except requests.exceptions.RequestException as e:
        print(f"错误：请求ArXiv页面失败: {e}")
        return None

    title_from_page = soup.find('h1', class_='title').text.replace('Title:', '').strip()

    # --- 3. 调用Semantic Scholar API获取丰富信息 ---
    print(f"正在调用Semantic Scholar API获取指定信息...")
    api_fields = [
        'citationCount', 'influentialCitationCount',
        'authors.name', 'authors.affiliations',
        'venue', 'publicationVenue', 'publicationDate',
        'fieldsOfStudy'
    ]
    api_url = f'https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}'
    api_params = {'fields': ','.join(api_fields)}
    
    paper_info = {}
    try:
        api_response = requests.get(api_url, headers=headers, params=api_params, timeout=10)
        api_response.raise_for_status()
        api_data = api_response.json()

        authors_list = [author.get('name', '未知作者') for author in api_data.get('authors', [])]
        institutions = set()
        if api_data.get('authors'):
            for author in api_data['authors']:
                if author.get('affiliations'):
                    # 过滤掉空的机构信息
                    institutions.update(aff for aff in author['affiliations'] if aff and aff.strip())

        paper_info = {
            'link': arxiv_url,
            'pdf_link': arxiv_url.replace('/abs/', '/pdf/'),
            'title': title_from_page,
            'authors': authors_list,
            'institutions': sorted(list(institutions)) if institutions else ["未找到机构信息"],
            'publication_date': api_data.get('publicationDate', "未找到"),
            'venue': api_data.get('venue') or (api_data.get('publicationVenue') or {}).get('name') or "未找到",
            'fields_of_study': api_data.get('fieldsOfStudy', []),
            'citation_count': api_data.get('citationCount', 0),
            'influential_citation_count': api_data.get('influentialCitationCount', 0),
        }

    except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
        print(f"警告：调用或解析Semantic Scholar API失败: {e}。无法生成 {arxiv_id} 的元数据。")
        return None # <-- 重要修改：返回None而不是raise，以防中断批处理
        
    # --- 4. 将字典转换为格式化的JSON字符串 ---
    return json.dumps(paper_info, indent=4, ensure_ascii=False)

# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 1. 设置文件路径 ---
    # 使用Pathlib进行路径管理，更清晰、跨平台
    base_path = Path("/home/zhangping/jrz-test/PaperSAGE/1.data_capture")
    date_folder = "extra"
    md_folder = base_path / "transferred_papers" / date_folder
    db_folder = base_path / "database" / date_folder

    # 确保输出目录存在，如果不存在则创建
    db_folder.mkdir(parents=True, exist_ok=True)

    # --- 2. 获取所有md文件列表 ---
    md_files = list(md_folder.glob("*.md"))

    if not md_files:
        print(f"在目录 {md_folder} 中没有找到任何.md文件。程序退出。")
    else:
        print(f"发现 {len(md_files)} 个.md文件。开始检查并处理新文件...")

    # --- 3. 遍历MD文件，检查对应的JSON是否存在，若不存在则处理 ---
    for md_path in md_files:
        # 从MD文件路径中提取arxiv_id (例如 '2404.13076')
        arxiv_id = md_path.stem
        
        # 构建对应的JSON文件路径
        json_path = db_folder / f"{arxiv_id}.json"

        # 核心逻辑：检查JSON文件是否已存在
        if json_path.exists():
            print(f"跳过: {arxiv_id}.json 已存在。")
            continue  # 跳过当前循环，处理下一个文件
        
        # --- 如果JSON文件不存在，执行完整处理流程 ---
        print(f"\n--- 发现新文件，开始处理: {arxiv_id} ---")
        
        # 步骤 A: 爬取元数据
        target_url = f"https://arxiv.org/abs/{arxiv_id}"
        json_output = scrape_arxiv_paper(target_url)

        # 如果爬取失败（函数返回None），则跳过此文件
        if not json_output:
            print(f"处理失败: 无法获取 {arxiv_id} 的元数据。跳过此文件。")
            continue
            
        paper_data_dict = json.loads(json_output)
        print(paper_data_dict)

        # 步骤 B: 读取MD文件内容
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                paper_content = f.read()
            paper_data_dict['paper_content'] = paper_content
        except FileNotFoundError:
            print(f"错误: 找不到MD文件 {md_path}。跳过此文件。")
            continue

        # 步骤 C: 生成摘要
        final_summary = summary_func(arxiv_id)
        paper_data_dict['summary'] = final_summary
        
        # 步骤 D: 保存最终的JSON文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(paper_data_dict, f, ensure_ascii=False, indent=4)
        
        print(f"成功: {arxiv_id}.json 已创建并保存。")

    print("\n所有文件处理完毕。")
