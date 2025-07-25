import re
import os
import glob
import numpy as np
from datetime import datetime
from transformers import AutoTokenizer
from prompts.summary_prompt import summary_prompt_v3
import logging
import sys

# --- 0. 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- 1. 全局配置区 ---
now = datetime.now()
formatted_date = now.strftime("%Y%m%d")
# 存放待处理 PDF 文件的文件夹路径
# INPUT_FOLDER = f"origin_papers/{formatted_date}"
INPUT_FOLDER = "origin_papers/aaai-technical-track-on-natural-language-processing-iii"
# 存放解析后 Markdown 文件的文件夹路径
# OUTPUT_FOLDER = f"transferred_papers/{formatted_date}"
OUTPUT_FOLDER = "transferred_papers/extra"
TOKENIZER_PATH = "/home/zhangping/jrz-test/models/sft/Qwen/Qwen2.5-7B"


# --- 脚本主体 ---
def remove_unwanted_sections(markdown_content, section_titles):
    """
    从Markdown文本中移除指定的章节及其之后的所有内容。
    """
    cut_off_pos = len(markdown_content)

    for title in section_titles:
        pattern_str = f"^#?\\s*(\\d+\\.?\\s+)?{re.escape(title)}\\b"
        section_pattern = re.compile(pattern_str, re.MULTILINE)
        match = section_pattern.search(markdown_content)

        if match:
            if match.start() < cut_off_pos:
                cut_off_pos = match.start()

    return markdown_content[:cut_off_pos].strip()


def process_markdown_files(md_folder, pdf_folder, tokenizer_path, summary_prompt_template, print_statics=False):
    """
    处理指定文件夹中所有的Markdown文件，并删除对应的PDF。
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        logger.error(f"加载Tokenizer '{tokenizer_path}'失败: {e}", exc_info=True)
        return []
        
    md_files = glob.glob(os.path.join(md_folder, "*.md"))
    
    sections_to_remove = ["Acknowledgement", "Acknowledgements", "Reference", "References", "Acknowledgment", "Acknowledgments",
                          "ACKNOWLEDGEMENT", "ACKNOWLEDGEMENTS", "REFERENCE", "REFERENCES", "ACKNOWLEDGMENT", "ACKNOWLEDGMENTS"]
    
    results = []
    token_counts = []

    if not md_files:
        logger.info(f"在文件夹 '{md_folder}' 中未找到任何Markdown文件。")
        return results
        
    for md_file_path in md_files:
        filename = os.path.basename(md_file_path)
        base_name = os.path.splitext(filename)[0]
        
        # --- 删除对应的PDF文件 ---
        pdf_to_delete = os.path.join(pdf_folder, base_name + '.pdf')
        if os.path.exists(pdf_to_delete):
            try:
                os.remove(pdf_to_delete)
                logger.info(f"已找到并删除对应的PDF文件: {os.path.basename(pdf_to_delete)}")
            except OSError as e:
                logger.error(f"删除PDF文件 {os.path.basename(pdf_to_delete)} 时出错: {e}")
        
        # --- 处理Markdown文件 ---
        try:
            with open(md_file_path, 'r', encoding='utf-8') as f:
                paper_content = f.read()
            
            cleaned_paper = remove_unwanted_sections(paper_content, sections_to_remove)
            
            with open(md_file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_paper)
            
            summary_prompt = summary_prompt_template.format(paper=cleaned_paper)
            token_count = len(tokenizer.encode(summary_prompt))
            token_counts.append(token_count)
            
            results.append({
                "file": filename,
                "token_count": token_count,
            })
            
            logger.info(f"已处理并覆盖保存 {filename} - 令牌数: {token_count}")

        except Exception as e:
            logger.error(f"处理文件 {filename} 时发生错误: {e}", exc_info=True)

    # --- 打印统计数据 ---
    if token_counts and print_statics:
        token_counts_np = np.array(token_counts)
        mean = np.mean(token_counts_np)
        max_val = np.max(token_counts_np)
        min_val = np.min(token_counts_np)
        std_dev = np.std(token_counts_np)
        mean_plus_3std = mean + 3 * std_dev
        
        logger.info("\n--- 统计信息 ---")
        logger.info(f"处理文件总数: {len(token_counts)}")
        logger.info(f"平均 Token 数: {mean:.2f}")
        logger.info(f"最大 Token 数: {max_val}")
        logger.info(f"最小 Token 数: {min_val}")
        logger.info(f"平均值 + 3倍标准差: {mean_plus_3std:.2f}")
        
        outlier_count = np.sum(token_counts_np > mean_plus_3std)
        logger.info(f" Token 数超过 '均值+3倍标准差' 的文件数量: {outlier_count}")
        if len(token_counts_np) > 0:
            proportion = outlier_count / len(token_counts_np)
            logger.info(f"该部分文件占比: {proportion:.2%}")

    return results

if __name__ == "__main__":
    results = process_markdown_files(
        md_folder=OUTPUT_FOLDER,
        pdf_folder=INPUT_FOLDER,
        tokenizer_path=TOKENIZER_PATH,
        summary_prompt_template=summary_prompt_v3,
        print_statics=True
    )
