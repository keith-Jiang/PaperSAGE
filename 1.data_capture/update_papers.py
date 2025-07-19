import os
import arxiv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import sys
import re

# --- 1. 配置区 ---
CATEGORIES = [
    "cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.RO", 
    "cs.NE", "cs.IR", "cs.DC", "cs.AR", "cs.CR"
]
MAX_PAPERS = 10
now = datetime.now()
formatted_date = now.strftime("%Y%m%d")
PAPERS_DIR = Path(f"./origin_papers/{formatted_date}")

# --- 脚本主体 ---

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# 如果不存在论文目录则创建 (确保父目录也能被创建)
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"论文将保存在: {PAPERS_DIR.absolute()}")

def sanitize_filename(name: str) -> str:
    """
    清理字符串，使其成为一个有效且安全的文件名。

    - 移除Windows和Linux/macOS中的非法字符。
    - 将空格替换为下划线。
    - 限制文件名长度以避免路径过长问题。
    """
    # 移除或替换文件名中的非法字符
    sanitized_name = re.sub(r'[<>:"/\\|?*]', '', name)
    # 将多个空格替换为单个下划线
    sanitized_name = re.sub(r'\s+', '_', sanitized_name)
    # 限制文件名长度，避免过长 (文件名主体部分)
    max_len = 180
    if len(sanitized_name) > max_len:
        sanitized_name = sanitized_name[:max_len]
    return sanitized_name.strip('._') # 移除开头和结尾的下划线或点

def get_daily_papers(categories, max_results=MAX_PAPERS):
    """
    获取最近3天内发布的指定类别的论文，以应对ArXiv的发布周期。
    """
    # (此函数保持不变)
    now_utc = datetime.now(timezone.utc)
    three_days_ago_utc = now_utc - timedelta(days=3)
    start_date = three_days_ago_utc.strftime('%Y%m%d%H%M%S')
    end_date = now_utc.strftime('%Y%m%d%H%M%S')
    category_query = " OR ".join([f"cat:{cat}" for cat in categories])
    date_query = f"submittedDate:[{start_date} TO {end_date}]"
    query = f"({category_query}) AND {date_query}"
    
    logger.info(f"正在向ArXiv API发送精确查询: {query}")
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )
    results = list(client.results(search))
    
    logger.info(f"通过API查询在过去3天内找到 {len(results)} 篇符合条件的论文")
    return results

def download_paper(paper, output_dir):
    """将论文PDF下载到指定目录，使用论文标题作为文件名。"""
    # 1. 清理论文标题，生成安全的文件名
    safe_title = sanitize_filename(paper.title)
    # 2. 拼接成完整的 .pdf 文件名
    pdf_filename = f"{safe_title}.pdf"
    
    pdf_path = output_dir / pdf_filename
    
    if pdf_path.exists():
        logger.info(f"论文已存在，跳过下载: {paper.title}")
        return
    
    try:
        logger.info(f"开始下载: {paper.title}")
        # 使用新生成的文件名进行下载
        paper.download_pdf(dirpath=str(output_dir), filename=pdf_filename)
        logger.info(f"下载成功 -> {pdf_path.name}")
    except Exception as e:
        logger.error(f"下载失败 {paper.title}: {str(e)}")

def main():
    logger.info("开始每日ArXiv论文下载任务")
    papers = get_daily_papers(CATEGORIES, MAX_PAPERS)
    
    if not papers:
        logger.info("过去3天内没有找到新论文。任务结束。")
        return
    
    logger.info(f"准备下载 {len(papers)} 篇论文...")
    for i, paper in enumerate(papers, 1):
        logger.info(f"--- 正在处理第 {i}/{len(papers)} 篇 ---")
        download_paper(paper, PAPERS_DIR)
    
    logger.info("所有新论文下载完成！")

if __name__ == "__main__":
    main()
