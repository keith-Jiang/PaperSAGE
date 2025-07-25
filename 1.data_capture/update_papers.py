import os
import arxiv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
import sys
import re

# --- 0. 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- 1. 全局配置区 ---
CATEGORIES = [
    "cs.AI", "cs.AR", "cs.CC", "cs.CE", "cs.CG", "cs.CL", "cs.CR", "cs.CV", "cs.CY", 
    "cs.DB", "cs.DC", "cs.DL", "cs.DM", "cs.DS", "cs.ET", "cs.FL", "cs.GL", "cs.GR",
    "cs.GT", "cs.HC", "cs.IR", "cs.IT", "cs.LG", "cs.LO", "cs.MA", "cs.MM", "cs.MS",
    "cs.NA", "cs.NE", "cs.NI", "cs.OH", "cs.OS", "cs.PF", "cs.PL", "cs.RO", "cs.SC",
    "cs.SD", "cs.SE", "cs.SI", "cs.SY"
]
MAX_PAPERS = 50
now = datetime.now()
formatted_date = now.strftime("%Y%m%d")
PAPERS_DIR = Path(f"./origin_papers/{formatted_date}")

# --- 脚本主体 ---
# 如果不存在论文目录则创建 (确保父目录也能被创建)
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"论文将保存在: {PAPERS_DIR.absolute()}")

def get_daily_papers(categories, max_results=MAX_PAPERS):
    """
    获取最近3天内发布的指定类别的论文，以应对ArXiv的发布周期。
    """
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
    """将论文PDF下载到指定目录，使用ArXiv ID作为文件名。"""
    # 1. 获取论文的ArXiv ID作为文件名 (例如 '2301.12345v1')
    arxiv_id = paper.get_short_id()
    arxiv_id = arxiv_id.split('v')[0]  # 移除版本后缀
    pdf_filename = f"{arxiv_id}.pdf"
    
    pdf_path = output_dir / pdf_filename
    
    if pdf_path.exists():
        logger.info(f"论文 '{pdf_path.name}' 已存在，跳过下载: {paper.title}")
        return
    
    try:
        logger.info(f"开始下载: {paper.title} (ID: {arxiv_id})")
        # 使用新生成的文件名进行下载
        paper.download_pdf(dirpath=str(output_dir), filename=pdf_filename)
        logger.info(f"下载成功 -> {pdf_path.name}")
    except Exception as e:
        logger.error(f"下载失败 {paper.title} (ID: {arxiv_id}): {str(e)}")

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
