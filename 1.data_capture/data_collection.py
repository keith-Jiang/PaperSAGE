import requests
import json
import re
import time
from datetime import datetime
from bs4 import BeautifulSoup
from summary_test_api import summary_func
from pathlib import Path
import os
from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark

def scrape_arxiv_paper(arxiv_url):
    """
    爬取指定的ArXiv论文页面，提取核心信息。
    优先使用Semantic Scholar API获取丰富信息，如果API返回404（未收录），
    则降级使用从ArXiv页面直接爬取的基础信息。包含重试机制。

    :param arxiv_url: 论文的ArXiv链接 (例如: 'https://arxiv.org/abs/1706.03762')
    :return: 包含论文信息的JSON字符串，如果发生无法恢复的错误则返回None
    """
    
    # --- 1. 准备工作 ---
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    max_retries = 3
    retry_delay = 5  # seconds

    try:
        # 正则表达式可以更稳定地提取ID，兼容如 /pdf/1706.03762.pdf 这样的链接
        match = re.search(r'(\d{4}\.\d{4,5})', arxiv_url)
        if not match:
            print("错误：无法从URL中解析ArXiv ID。请检查URL格式。")
            return None
        arxiv_id = match.group(1)
        # 确保URL是/abs/格式
        abs_url = f'https://arxiv.org/abs/{arxiv_id}'
        pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'

    except (IndexError, AttributeError):
        print("错误：无法从URL中解析ArXiv ID。请检查URL格式。")
        return None

    # --- 2. 爬取ArXiv页面，获取基础信息作为备用 ---
    print(f"正在从ArXiv爬取基础信息: {abs_url}")
    try:
        response = requests.get(abs_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        # 提取备用信息
        title_from_page = soup.find('h1', class_='title').text.replace('Title:', '').strip()
        authors_from_page = [a.text.strip() for a in soup.select('.authors a')]
        # 提取提交日期
        dateline_text = soup.find('div', class_='dateline').text
        date_match = re.search(r'\(Submitted on (.+?)\)', dateline_text)
        date_from_page = date_match.group(1) if date_match else "未找到提交日期"

    except requests.exceptions.RequestException as e:
        print(f"错误：请求ArXiv页面失败: {e}")
        return None
    except AttributeError:
        # 如果页面结构有变，导致find/select失败
        print(f"错误：解析ArXiv页面HTML结构失败，可能页面已更新。")
        return None


    # --- 3. 调用Semantic Scholar API获取丰富信息（带重试机制） ---
    print(f"正在调用Semantic Scholar API获取 {arxiv_id} 的补充信息...")
    api_fields = [
        'citationCount', 'influentialCitationCount',
        'authors.name',
        'venue', 'publicationVenue', 'publicationDate',
        'fieldsOfStudy'
    ]
    api_url = f'https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}'
    api_params = {'fields': ','.join(api_fields)}
    
    paper_info = {}
    
    for attempt in range(max_retries):
        try:
            api_response = requests.get(api_url, headers=headers, params=api_params, timeout=15)
            # raise_for_status() 会对4xx和5xx状态码抛出HTTPError
            api_response.raise_for_status()
            
            api_data = api_response.json()
            authors_list = [author.get('name', '未知作者') for author in api_data.get('authors', [])]

            paper_info = {
                'source': 'Semantic Scholar',
                'arxiv_id': arxiv_id,
                'link': abs_url,
                'pdf_link': pdf_url,
                'title': api_data.get('title', title_from_page), # 优先用API的，其次用页面的
                'authors': authors_list,
                'publication_date': api_data.get('publicationDate', "未找到发表日期"),
                'venue': api_data.get('venue') or (api_data.get('publicationVenue') or {}).get('name') or "未找到发表会议",
                'fields_of_study': api_data.get('fieldsOfStudy', "未找到研究领域"),
                'citation_count': api_data.get('citationCount', "未找到引用量"),
                'influential_citation_count': api_data.get('influentialCitationCount', "未找到高影响力引用量"),
            }
            print("成功从Semantic Scholar获取信息。")
            break  # 成功获取，跳出重试循环

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"警告：Semantic Scholar API返回404，说明论文 {arxiv_id} 暂未被收录。")
                # 使用从ArXiv页面爬取的基础信息
                paper_info = {
                    'source': 'ArXiv (Semantic Scholar未收录)',
                    'arxiv_id': arxiv_id,
                    'link': abs_url,
                    'pdf_link': pdf_url,
                    'title': title_from_page,
                    'authors': authors_from_page,
                    'publication_date': date_from_page, # 使用ArXiv的提交日期
                    'venue': "暂未录入Semantic Scholar",
                    'fields_of_study': "暂未录入Semantic Scholar",
                    'citation_count': "暂未录入Semantic Scholar",
                    'influential_citation_count': "暂未录入Semantic Scholar",
                }
                break # 确定是404，无需重试，跳出循环
            else:
                # 其他HTTP错误 (如 500, 503, 403)
                print(f"错误：API请求失败，状态码: {e.response.status_code}。尝试 {attempt + 1}/{max_retries}...")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    print("已达到最大重试次数，API请求失败。")
                    return None  # 所有重试都失败后返回None

        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            # 其他网络错误 (如超时) 或JSON解析错误
            print(f"错误：API请求或解析失败: {e}。尝试 {attempt + 1}/{max_retries}...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("已达到最大重试次数，API请求失败。")
                return None  # 所有重试都失败后返回None

    # --- 4. 将字典转换为格式化的JSON字符串 ---
    if not paper_info:
        # 如果循环结束，paper_info仍然是空的，说明所有尝试都失败了
        print("最终未能获取到论文信息。")
        return None
        
    return json.dumps(paper_info, indent=4, ensure_ascii=False)

def extract_institutions_with_ai(paper_content, client, model_name):
    print("正在调用AI模型提取机构信息...")
    
    # 1. 精准提取第一个和第二个'#'之间的内容
    parts = paper_content.split('#')
    authors_and_affiliations_text = parts[1].strip()

    # 2. 构建Prompt
    prompt = f"""
    请从以下文本中仔细抽取出所有的独立机构名称（university, institute, lab, company, etc.）。
    要求：
    1.  每个机构名称单独占一行。
    2.  不要添加任何序号、破折号、星号或其他标记。
    3.  不要包含任何解释性文字或标题，如“提取的机构如下：”。
    4.  确保机构名称的完整性和准确性，合并同一机构的不同表述（如 "UC Berkeley" 和 "University of California, Berkeley" 应视为一个）。
    5.  如果文本中没有明确的机构信息，则返回"未找到机构信息"。

    待处理文本如下：
    ---
    {authors_and_affiliations_text}
    ---
    """

    # 3. 调用API
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个精准的信息提取助手，严格按照指令格式输出结果。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0, # 使用低温以获得更稳定、确定性的输出
        )
        response_text = completion.choices[0].message.content
        
        # 4. 解析结果
        if not response_text or response_text.strip() == "":
            print("AI模型返回为空，未提取到机构信息。")
            return []
            
        # 按换行符分割，去除每个机构名称前后的空白，并过滤掉空行
        institutions = [line.strip() for line in response_text.split('\n') if line.strip()]
        print(f"机构生成完成: {institutions}")
        
        return institutions
    except Exception as e:
        print(f"错误: 调用AI模型API失败: {e}")
        return []

# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 1. 设置文件路径 ---
    # 使用Pathlib进行路径管理，更清晰、跨平台
    base_path = Path("/home/zhangping/jrz-test/PaperSAGE/1.data_capture")
    # now = datetime.now()
    # date_folder = now.strftime("%Y%m%d")
    date_folder = "20250721"
    md_folder = base_path / "transferred_papers" / date_folder
    db_folder = base_path / "database" / date_folder

    load_dotenv()
    ARK_MODEL = os.getenv("ARK_MODEL")
    ARK_KEY = os.getenv("ARK_API_KEY")
    BASE_URL = os.getenv("ARK_API_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3/")
    TIMEOUT = int(os.getenv("ARK_TIMEOUT", 1800))
    client = Ark(api_key=ARK_KEY, base_url=BASE_URL, timeout=TIMEOUT)

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
            time.sleep(8)
            continue

        paper_data_dict = json.loads(json_output)
        
        # 步骤 B: 获取机构
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                paper_content = f.read()
            paper_data_dict['paper_content'] = paper_content
        except FileNotFoundError:
            print(f"错误: 找不到MD文件 {md_path}。跳过此文件。")
            continue

        institutions = extract_institutions_with_ai(paper_content, client, ARK_MODEL)
        paper_data_dict['institutions'] = institutions
        
        # 步骤 C: 生成摘要
        final_summary = summary_func(arxiv_id, md_folder)
        paper_data_dict['summary'] = final_summary
        
        # 步骤 D: 保存最终的JSON文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(paper_data_dict, f, ensure_ascii=False, indent=4)
        
        print(f"成功: {arxiv_id}.json 已创建并保存。")
        time.sleep(8)

    print("\n所有文件处理完毕。")
