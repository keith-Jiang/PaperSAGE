import re
import os
import glob
import numpy as np
from transformers import AutoTokenizer
from prompts.summary_prompt import summary_prompt_v3

def remove_unwanted_sections(markdown_content, section_titles):
    """
    从Markdown文本中移除指定的章节及其之后的所有内容。

    这个函数会遍历一个章节标题列表，找到这些章节在文本中首次出现的位置，
    并从最早出现的那个章节标题开始，删除其及之后的所有内容。

    只匹配以下三种模式，且标题单词必须首字母大写：
    1. 标题前有单个井号(#)和空格/换行: "# Acknowledgement", "# \n\n Reference"
    2. 标题前有单个井号(#)和数字: "# 7. Acknowledgements", "# 8 Reference"
    3. 标题前无井号: "References" (如果以行首的非井号形式出现)

    参数:
        markdown_content (str): 输入的Markdown文件内容。
        section_titles (list): 一个包含要移除的章节标题字符串的精确列表（首字母大写）。

    返回:
        str: 移除了指定章节及其后所有内容的新Markdown内容。
    """
    cut_off_pos = len(markdown_content)

    for title in section_titles:
        pattern_str = f"^#?\\s*(\\d+\\.?\\s+)?{re.escape(title)}\\b"
        section_pattern = re.compile(pattern_str, re.MULTILINE)
        match = section_pattern.search(markdown_content)

        if match:
            # 找到最早出现的那个标题的位置
            if match.start() < cut_off_pos:
                cut_off_pos = match.start()

    # 如果找到了标题，则从该位置截断；否则不改变内容
    return markdown_content[:cut_off_pos].strip()


def process_markdown_files(md_folder, pdf_folder, tokenizer_path, summary_prompt_template):
    """
    处理指定文件夹中所有的Markdown文件，并删除对应的PDF。
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    md_files = glob.glob(os.path.join(md_folder, "*.md"))
    
    # 使用您指定的精确标题列表
    sections_to_remove = ["Acknowledgement", "Acknowledgements", "Reference", "References"]
    
    results = []
    token_counts = []

    if not md_files:
        print(f"在文件夹 '{md_folder}' 中未找到任何Markdown文件。")
        return results
        
    for md_file_path in md_files:
        filename = os.path.basename(md_file_path)
        base_name = os.path.splitext(filename)[0]
        
        # --- 删除对应的PDF文件 ---
        pdf_to_delete = os.path.join(pdf_folder, base_name + '.pdf')
        if os.path.exists(pdf_to_delete):
            try:
                os.remove(pdf_to_delete)
                print(f"已找到并删除对应的PDF文件: {os.path.basename(pdf_to_delete)}")
            except OSError as e:
                print(f"删除PDF文件 {os.path.basename(pdf_to_delete)} 时出错: {e}")
        
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
            
            print(f"已处理并覆盖保存 {filename} - 令牌数: {token_count}")

        except Exception as e:
            print(f"处理文件 {filename} 时发生错误: {e}")

    # 打印统计数据
    if token_counts:
        token_counts_np = np.array(token_counts)
        mean = np.mean(token_counts_np)
        max_val = np.max(token_counts_np)
        min_val = np.min(token_counts_np)
        std_dev = np.std(token_counts_np)
        mean_plus_3std = mean + 3 * std_dev
        
        print("\n--- 统计信息 ---")
        print(f"处理文件总数: {len(token_counts)}")
        print(f"平均 Token 数: {mean:.2f}")
        print(f"最大 Token 数: {max_val}")
        print(f"最小 Token 数: {min_val}")
        print(f"平均值 + 3倍标准差: {mean_plus_3std:.2f}")
        
        outlier_count = np.sum(token_counts_np > mean_plus_3std)
        print(f" Token 数超过 '均值+3倍标准差' 的文件数量: {outlier_count}")
        if len(token_counts_np) > 0:
            proportion = outlier_count / len(token_counts_np)
            print(f"该部分文件占比: {proportion:.2%}")

    return results

# --- 主程序入口 ---
if __name__ == "__main__":
    md_folder_path = "./transferred_papers"
    pdf_folder_path = "./origin_papers/accept-oral"
    tokenizer_path = "/home/zhangping/jrz-test/models/sft/Qwen/Qwen2.5-7B"
    
    if not os.path.isdir(pdf_folder_path):
        print(f"警告：PDF文件夹 '{pdf_folder_path}' 不存在，将跳过PDF删除步骤。")
    
    results = process_markdown_files(
        md_folder=md_folder_path,
        pdf_folder=pdf_folder_path,
        tokenizer_path=tokenizer_path,
        summary_prompt_template=summary_prompt_v3
    )

