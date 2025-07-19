import requests  # 导入用于发送HTTP请求的库
import json      # 导入用于处理JSON数据的库
import re        # 导入正则表达式库，用于从文本中提取年份
from bs4 import BeautifulSoup  # 导入BeautifulSoup，用于解析HTML

def scrape_arxiv_paper(arxiv_url):
    """
    爬取指定的ArXiv论文页面，提取信息并以JSON格式返回。

    :param arxiv_url: 论文的ArXiv链接 (例如: 'https://arxiv.org/abs/1706.03762')
    :return: 包含论文信息的JSON字符串，如果失败则返回None
    """
    
    # --- 1. 准备工作：设置请求头，并从URL中提取ArXiv ID ---
    
    # 伪装成浏览器发送请求，避免被服务器阻止
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    # 从URL中提取论文的唯一ID (例如 '1706.03762')
    # 这是调用API和后续处理的关键
    try:
        arxiv_id = arxiv_url.split('/')[-1]
    except IndexError:
        print("错误：无法从URL中解析ArXiv ID。请检查URL格式。")
        return None

    # --- 2. 爬取ArXiv页面本身，获取基本信息 ---
    
    print(f"正在从ArXiv爬取基础信息: {arxiv_url}")
    try:
        # 发送GET请求获取网页内容
        response = requests.get(arxiv_url, headers=headers)
        # 如果请求不成功（状态码不是200），则抛出异常
        response.raise_for_status()  
    except requests.exceptions.RequestException as e:
        print(f"错误：请求ArXiv页面失败: {e}")
        return None

    # 使用BeautifulSoup和lxml解析器解析HTML内容
    soup = BeautifulSoup(response.text, 'lxml')
    print("66666")
    # --- 3. 从ArXiv页面提取信息 ---

    # 提取标题
    # 标题通常在<h1 class="title mathjax">标签中，我们去掉前面的'Title:'文本
    title_tag = soup.find('h1', class_='title')
    title = title_tag.text.replace('Title:', '').strip() if title_tag else "未找到标题"

    # 提取年份
    # 年份信息在<div class="dateline">中，格式如'(Submitted on 6 Jun 2017)'
    dateline_tag = soup.find('div', class_='dateline')
    year = "未找到年份"
    if dateline_tag:
        # 使用正则表达式匹配文本中的四位数字（年份）
        match = re.search(r'\b(\d{4})\b', dateline_tag.text)
        if match:
            year = match.group(1)
    print("66666")
    # --- 4. 调用Semantic Scholar API获取引用、作者和机构信息 ---
    # ArXiv本身不提供引用数，作者机构信息格式不统一，因此调用API是最佳选择
    
    print(f"正在调用Semantic Scholar API获取引用、作者和机构信息...")
    api_url = f'https://api.semanticscholar.org/v1/paper/arXiv:{arxiv_id}'
    
    authors_list = []
    institutions = set()  # 使用集合来存储机构，自动去重
    citation_count = 0

    try:
        api_response = requests.get(api_url, headers=headers)
        api_response.raise_for_status()
        api_data = api_response.json()

        # 提取引用量
        citation_count = api_data.get('citationCount', 0)

        # 提取作者和机构信息
        # API返回的作者信息更结构化，包含机构
        if 'authors' in api_data:
            for author in api_data['authors']:
                # 将作者名添加到作者列表
                authors_list.append(author.get('name', '未知作者'))
                # 如果作者有关联机构，提取并添加到机构集合中
                if author.get('affiliations'):
                    for affiliation in author.get('affiliations'):
                        if affiliation: # 确保机构名不为空
                            institutions.add(affiliation)
        print("66666")
    except requests.exceptions.RequestException as e:
        print(f"警告：请求Semantic Scholar API失败: {e}。引用、作者和机构信息可能不完整。")
        # 如果API失败，我们仍然可以从ArXiv页面获取作者列表作为备用方案
        authors_tag = soup.find('div', class_='authors')
        if authors_tag:
            authors_list = [a.text.strip() for a in authors_tag.find_all('a')]
    
    except json.JSONDecodeError:
        print("警告：解析Semantic Scholar API响应失败。")

    # --- 5. 整合所有信息到字典中 ---

    paper_info = {
        'link': arxiv_url,
        'title': title,
        'authors': authors_list,
        'institutions': list(institutions) if institutions else ["未找到机构信息"], # 将集合转为列表
        'year': year,
        'citation_count': citation_count,
    }

    # --- 6. 将字典转换为格式化的JSON字符串 ---

    # json.dumps用于将python字典转换为JSON字符串
    # indent=4: 使JSON输出格式化，有4个空格的缩进，更易读
    # ensure_ascii=False: 允许输出非ASCII字符（如中文），而不是转义成\uXXXX格式
    return json.dumps(paper_info, indent=4, ensure_ascii=False)


# --- 主程序入口 ---
if __name__ == "__main__":
    # 示例：使用著名的"Attention Is All You Need"论文的URL
    # 你可以替换成任何你想要的ArXiv论文链接
    # target_url = 'https://arxiv.org/abs/1706.03762' # Transformer
    target_url = 'https://arxiv.org/abs/1512.03385' # ResNet

    print(f"开始爬取论文: {target_url}\n")
    
    # 调用爬虫函数
    json_output = scrape_arxiv_paper(target_url)

    # 如果成功获取到数据，则打印输出
    if json_output:
        print("\n--- 爬取结果 (JSON格式) ---\n")
        print(json_output)
