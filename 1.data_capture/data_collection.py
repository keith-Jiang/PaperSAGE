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
import logging
import sys

# --- 0. 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- 1. 全局配置区 ---
# 基础工作路径
BASE_PATH = Path("/home/zhangping/jrz-test/PaperSAGE/1.data_capture")
# 根据当前日期或指定日期生成文件夹名称
# now = datetime.now()
# DATE_FOLDER = now.strftime("%Y%m%d")
DATE_FOLDER = "20250722" # 固定日期，方便测试
# 存放待处理的Markdown文件的文件夹路径
MD_FOLDER = BASE_PATH / "transferred_papers" / DATE_FOLDER
# 存放最终生成的JSON数据库文件的文件夹路径
DB_FOLDER = BASE_PATH / "database" / DATE_FOLDER
# 加载环境变量 (需要一个.env文件在项目根目录，包含API密钥等)
load_dotenv()
# AI模型及API相关配置
ARK_MODEL = os.getenv("ARK_MODEL")
ARK_KEY = os.getenv("ARK_API_KEY")
BASE_URL = os.getenv("ARK_API_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3/")
TIMEOUT = int(os.getenv("ARK_TIMEOUT", 1800))
# 初始化AI模型客户端
CLIENT = Ark(api_key=ARK_KEY, base_url=BASE_URL, timeout=TIMEOUT)
# 确保输出目录存在，如果不存在则创建
DB_FOLDER.mkdir(parents=True, exist_ok=True)

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
    max_retries = 3  # 最大重试次数
    retry_delay = 5  # 重试间隔时间（秒）

    try:
        match = re.search(r'(\d{4}\.\d{4,5})', arxiv_url)
        if not match:
            logger.error("无法从URL中解析ArXiv ID。请检查URL格式。")
            return None
        arxiv_id = match.group(1)
        abs_url = f'https://arxiv.org/abs/{arxiv_id}'
        pdf_url = f'https://arxiv.org/pdf/{arxiv_id}.pdf'

    except (IndexError, AttributeError):
        logger.error("无法从URL中解析ArXiv ID。请检查URL格式。")
        return None

    # --- 2. 爬取ArXiv页面，获取基础信息（包括分类）作为备用和补充 ---
    logger.info(f"正在从ArXiv爬取基础信息: {abs_url}")
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

        # 提取分类信息
        subjects_text = soup.find('td', class_='subjects').text.strip() if soup.find('td', class_='subjects') else ""
        # 使用正则表达式精确提取所有分类代码，如cs.AI, cs.LG等
        categories_from_page = re.findall(r'cs\.[A-Z]{2,}', subjects_text)
        if not categories_from_page: # 如果正则没匹配到，使用完整文本作为备用
             categories_from_page = [subjects_text] if subjects_text else ["未找到分类"]
        logger.info(f"从ArXiv页面提取到分类: {categories_from_page}")

    except requests.exceptions.RequestException as e:
        logger.error(f"请求ArXiv页面失败: {e}")
        return None
    except AttributeError:
        logger.error("解析ArXiv页面HTML结构失败，可能页面已更新。")
        return None

    # --- 3. 调用Semantic Scholar API获取丰富信息（带重试机制） ---
    logger.info(f"正在调用Semantic Scholar API获取 {arxiv_id} 的补充信息...")
    api_fields = ['citationCount', 'influentialCitationCount', 'authors.name', 'venue', 'publicationVenue', 'publicationDate', 'fieldsOfStudy']
    api_url = f'https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}'
    api_params = {'fields': ','.join(api_fields)}
    
    paper_info = {}
    
    for attempt in range(max_retries):
        try:
            api_response = requests.get(api_url, headers=headers, params=api_params, timeout=15)
            api_response.raise_for_status()
            
            api_data = api_response.json()
            authors_list = [author.get('name', '未知作者') for author in api_data.get('authors', [])]

            paper_info = {
                'source': 'Semantic Scholar',
                'arxiv_id': arxiv_id,
                'link': abs_url,
                'pdf_link': pdf_url,
                'title': api_data.get('title', title_from_page),
                'authors': authors_list,
                'categories': categories_from_page, # **新增**
                'publication_date': api_data.get('publicationDate', date_from_page),
                'venue': api_data.get('venue') or (api_data.get('publicationVenue') or {}).get('name') or "未找到发表会议",
                'fields_of_study': api_data.get('fieldsOfStudy', "未找到研究领域"),
                'citation_count': api_data.get('citationCount', "未找到引用量"),
                'influential_citation_count': api_data.get('influentialCitationCount', "未找到高影响力引用量"),
            }
            logger.info("成功从Semantic Scholar获取信息。")
            break

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                logger.warning(f"Semantic Scholar API返回404，论文 {arxiv_id} 暂未被收录。将使用ArXiv页面信息。")
                paper_info = {
                    'source': 'ArXiv (Semantic Scholar未收录)',
                    'arxiv_id': arxiv_id,
                    'link': abs_url,
                    'pdf_link': pdf_url,
                    'title': title_from_page,
                    'authors': authors_from_page,
                    'categories': categories_from_page, # **新增**
                    'publication_date': date_from_page,
                    'venue': "暂未录入Semantic Scholar",
                    'fields_of_study': "暂未录入Semantic Scholar",
                    'citation_count': "暂未录入Semantic Scholar",
                    'influential_citation_count': "暂未录入Semantic Scholar",
                }
                break
            else:
                logger.error(f"API请求失败，状态码: {e.response.status_code}。尝试 {attempt + 1}/{max_retries}...")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    logger.error("已达到最大重试次数，API请求失败。")
                    return None

        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            logger.error(f"API请求或解析失败: {e}。尝试 {attempt + 1}/{max_retries}...")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                logger.error("已达到最大重试次数，API请求失败。")
                return None

    # --- 4. 将字典转换为格式化的JSON字符串 ---
    if not paper_info:
        logger.error("最终未能获取到论文信息。")
        return None
        
    return json.dumps(paper_info, indent=4, ensure_ascii=False)

def extract_institutions_with_ai(paper_content, client, model_name):
    """
    使用AI模型从论文内容中提取机构信息。

    :param paper_content: 论文的完整Markdown内容字符串
    :param client: AI模型的客户端实例
    :param model_name: 要使用的AI模型名称
    :return: 包含机构名称的列表，如果失败则返回空列表
    """
    logger.info("正在调用AI模型提取机构信息...")
    
    try:
        parts = paper_content.split('#')
        authors_and_affiliations_text = parts[1].strip()
    except IndexError:
        logger.warning("Markdown内容格式不符合预期（未找到'#'分隔符），将尝试使用全文提取。")
        authors_and_affiliations_text = paper_content

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

    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个精准的信息提取助手，严格按照指令格式输出结果。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        response_text = completion.choices[0].message.content
        
        if not response_text or response_text.strip() == "" or "未找到机构信息" in response_text:
            logger.info("AI模型未提取到机构信息。")
            return ["未找到机构信息"]
            
        institutions = [line.strip() for line in response_text.split('\n') if line.strip()]
        logger.info(f"机构提取完成: {institutions}")
        
        return institutions
    except Exception as e:
        logger.error(f"调用AI模型API失败: {e}")
        return []

# --- 主程序入口 ---
if __name__ == "__main__":
    md_files = list(MD_FOLDER.glob("*.md"))

    if not md_files:
        logger.warning(f"在目录 {MD_FOLDER} 中没有找到任何.md文件。程序退出。")
    else:
        logger.info(f"发现 {len(md_files)} 个.md文件。开始处理...")

    for md_path in md_files:
        arxiv_id = md_path.stem
        logger.info(f"--- 开始检查文件: {arxiv_id} ---")
        # --- 步骤 A: 根据分类确定最终输出路径并检查文件是否存在 ---
        DB_FOLDER.mkdir(parents=True, exist_ok=True)
        json_path = DB_FOLDER / f"{arxiv_id}.json"

        if json_path.exists():
            logger.info(f"跳过: {json_path} 已存在。")
            continue

        logger.info(f"发现新论文 {arxiv_id}。开始完整处理...")

        # --- 步骤 B: 爬取元数据，以确定分类和最终路径 ---
        target_url = f"https://arxiv.org/abs/{arxiv_id}"
        json_output = scrape_arxiv_paper(target_url)

        if not json_output:
            logger.error(f"处理失败: 无法获取 {arxiv_id} 的元数据。跳过此文件。")
            continue
            
        paper_data_dict = json.loads(json_output)

        # 步骤 C: 读取MD文件内容并提取机构信息
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                paper_content = f.read()
        except FileNotFoundError:
            logger.error(f"找不到MD文件 {md_path}。跳过此文件。")
            continue

        institutions = extract_institutions_with_ai(paper_content, CLIENT, ARK_MODEL)
        paper_data_dict['institutions'] = institutions
        paper_data_dict['paper_content'] = paper_content
        
        # 步骤 D: 生成摘要
        final_summary = summary_func(md_path)
        paper_data_dict['summary'] = final_summary
        
        # 步骤 E: 保存最终的JSON文件到分类目录中
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(paper_data_dict, f, ensure_ascii=False, indent=4)
        
        logger.info(f"成功: {arxiv_id}.json 已创建并保存在 {json_path}")

    logger.info("所有文件处理完毕。")

