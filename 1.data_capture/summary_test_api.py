import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
import logging
import sys

from transformers import AutoTokenizer
from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark

# 假设这些prompt模块存在于指定的路径下
from prompts.ensemble_prompt import ensemble_prompt
from prompts.refine_prompt import refine_prompt
from prompts.summary_prompt import summary_prompt_v3

# --- 0. 日志配置 ---
# 配置日志记录器，设置级别为INFO，格式包含时间、级别和消息
# handlers指定将日志输出到标准输出流(控制台)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# --- 1. 全局配置与初始化 ---
load_dotenv()

# 从环境变量加载配置
ARK_MODEL = os.getenv("ARK_MODEL")
ARK_KEY = os.getenv("ARK_API_KEY")
BASE_URL = os.getenv("ARK_API_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3/")
TIMEOUT = int(os.getenv("ARK_TIMEOUT", 1800))
TOKENIZER_PATH = "/home/zhangping/jrz-test/models/sft/Qwen/Qwen2.5-7B"

try:
    client = Ark(api_key=ARK_KEY, base_url=BASE_URL, timeout=TIMEOUT)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    logger.info("Ark Client 和 Tokenizer 初始化成功。")
except Exception as e:
    logger.error(f"初始化服务时出错: {e}")
    sys.exit(1) # 初始化失败则退出程序

# --- 2. 抽离出通用的API调用函数 (遵循DRY原则) ---
def call_ark_api(prompt_content: str, temperature: float, top_p: float):
    """
    一个通用的函数，用于调用火山方舟大模型API。
    它接收格式化好的prompt，并返回模型的响应内容。
    """
    start_time = time.time()
    
    # 打印Token数量，用于调试和成本估算
    try:
        token_count = len(tokenizer.encode(prompt_content))
        logger.info(f"提交的 prompt token 数量为: {token_count} tokens...")
    except Exception as e:
        logger.warning(f"计算 token 数量时出错: {e}")

    try:
        completion = client.chat.completions.create(
            model=ARK_MODEL,
            messages=[{'role': 'user', 'content': prompt_content}],
            response_format={"type": "json_object"},
            temperature=temperature,
            top_p=top_p
        )
        content = completion.choices[0].message.content
        
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"API 调用成功。耗时: {total_time:.2f} 秒。")
        
        return content
    except Exception as e:
        logger.error(f"API 调用期间发生错误: {e}")
        return None # 在出错时返回None，方便上层处理

# --- 3. 主逻辑 ---
def summary_func(md_path):
    """
    生成论文摘要的主函数，包含发散、整合、反思三个阶段。
    
    :param arxiv_id: 论文的ArXiv ID
    :param data_folder: 存放论文Markdown文件的目录
    :return: 最终生成的摘要字符串，如果失败则返回None
    """
    # 读取论文文件
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            paper = f.read()
        logger.info(f"成功读取论文文件: {md_path}")
    except FileNotFoundError:
        logger.error(f"错误: 找不到论文文件 {md_path}。")
        return None

    summaries = {}

    # --- 阶段1：发散探索 (并行处理) ---
    logger.info("\n--- 阶段1：发散探索 (并行处理) ---")
    start_stage1_time = time.time()
    
    # 定义需要并行执行的任务
    tasks = {
        "summary_1": partial(call_ark_api, prompt_content=summary_prompt_v3.format(paper=paper), temperature=0.7, top_p=0.9),
        "summary_2": partial(call_ark_api, prompt_content=summary_prompt_v3.format(paper=paper), temperature=0.7, top_p=0.9),
    }

    # 使用线程池来并行执行API调用
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        # 提交所有任务
        future_to_name = {executor.submit(task_func): name for name, task_func in tasks.items()}
        
        # 等待任务完成并收集结果
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                result = future.result()
                if result:
                    summaries[name] = result
                    logger.info(f"成功完成任务: {name}")
                else:
                    logger.warning(f"任务 {name} 失败，未返回结果。")
            except Exception as exc:
                logger.error(f"任务 {name} 执行时产生异常: {exc}")

    end_stage1_time = time.time()
    logger.info(f"阶段1完成，耗时 {end_stage1_time - start_stage1_time:.2f} 秒。")
    
    # 检查必要摘要是否已生成
    if "summary_1" not in summaries or "summary_2" not in summaries:
        logger.error("未能生成初始摘要，处理中止。")
        return None

    logger.info(f"--- 初始摘要 1 ---\n{summaries['summary_1']}")
    logger.info(f"--- 初始摘要 2 ---\n{summaries['summary_2']}")

    # --- 阶段2：收敛整合 (串行处理) ---
    logger.info("\n--- 阶段2：收敛整合 ---")
    ensemble_prompt_formatted = ensemble_prompt.format(paper=paper, summary_1=summaries['summary_1'], summary_2=summaries['summary_2'])
    summary_3 = call_ark_api(ensemble_prompt_formatted, temperature=0.0, top_p=1.0)
    if not summary_3:
        logger.error("生成整合摘要失败，处理中止。")
        return None
    logger.info(f"--- 整合后摘要 ---\n{summary_3}")

    # --- 阶段3：收敛反思 (串行处理) ---
    logger.info("\n--- 阶段3：反思与精炼 ---")
    refine_prompt_formatted = refine_prompt.format(paper=paper, summary=summary_3)
    final_summary = call_ark_api(refine_prompt_formatted, temperature=0.0, top_p=1.0)
    if not final_summary:
        logger.error("生成最终精炼摘要失败。")
        return None
    logger.info(f"--- 最终反思摘要 ---\n{final_summary}")

    return final_summary


if __name__ == "__main__":
    pass
