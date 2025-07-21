import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from transformers import AutoTokenizer
from dotenv import load_dotenv
from volcenginesdkarkruntime import Ark

from prompts.ensemble_prompt import ensemble_prompt
from prompts.refine_prompt import refine_prompt
from prompts.summary_prompt import summary_prompt_v3

# --- 1. 全局配置与初始化 (只执行一次) ---
load_dotenv()

# 从环境变量加载配置
ARK_MODEL = os.getenv("ARK_MODEL")
ARK_KEY = os.getenv("ARK_API_KEY")
BASE_URL = os.getenv("ARK_API_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3/")
TIMEOUT = int(os.getenv("ARK_TIMEOUT", 1800))
TOKENIZER_PATH = "/home/zhangping/jrz-test/models/sft/Qwen/Qwen2.5-7B"

# 创建唯一的、可复用的客户端和分词器实例
# 这是性能优化的关键点之一：避免重复创建对象和建立连接
try:
    client = Ark(api_key=ARK_KEY, base_url=BASE_URL, timeout=TIMEOUT)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    print("Ark Client and Tokenizer initialized successfully.")
except Exception as e:
    print(f"Error initializing services: {e}")
    exit(1)

# --- 2. 抽离出通用的API调用函数 (遵循DRY原则) ---
def call_ark_api(prompt_content: str, temperature: float, top_p: float):
    """
    一个通用的函数，用于调用火山方舟大模型API。
    它接收格式化好的prompt，并返回模型的响应内容。
    """
    start_time = time.time()
    
    # 打印Token数量，用于调试和成本估算
    token_count = len(tokenizer.encode(prompt_content))
    print(f"Submitting prompt with {token_count} tokens...")
    
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
        print(f"API call successful. Time taken: {total_time:.2f} seconds.")
        
        return content
    except Exception as e:
        print(f"An error occurred during API call: {e}")
        return None # 在出错时返回None，方便上层处理

# --- 3. 主逻辑 ---
def summary_func(arxiv_id):
    # 读取论文文件
    try:
        with open(f"/home/zhangping/jrz-test/PaperSAGE/1.data_capture/transferred_papers/extra/{arxiv_id}.md", 'r', encoding='utf-8') as f:
            paper = f.read()
    except FileNotFoundError:
        print("Error: Paper file not found.")
        return

    summaries = {}

    # --- 阶段1：发散探索 (并行处理) ---
    print("\n--- Stage 1: Divergent Exploration (Running in Parallel) ---")
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
                    print(f"Successfully completed task: {name}")
                else:
                    print(f"Task {name} failed and returned no result.")
            except Exception as exc:
                print(f"Task {name} generated an exception: {exc}")

    end_stage1_time = time.time()
    print(f"Stage 1 completed in {end_stage1_time - start_stage1_time:.2f} seconds.")
    
    if "summary_1" not in summaries or "summary_2" not in summaries:
        print("Could not generate initial summaries. Aborting.")
        return

    print("--- 初始摘要 1 ---\n", summaries['summary_1'])
    print("--- 初始摘要 2 ---\n", summaries['summary_2'])

    # --- 阶段2：收敛整合 (串行处理) ---
    print("\n--- Stage 2: Convergent Integration ---")
    ensemble_prompt_formatted = ensemble_prompt.format(paper=paper, summary_1=summaries['summary_1'], summary_2=summaries['summary_2'])
    summary_3 = call_ark_api(ensemble_prompt_formatted, temperature=0.0, top_p=1.0)
    if not summary_3:
        print("Failed to generate ensemble summary. Aborting.")
        return
    print("--- 整合后摘要 ---\n", summary_3)

    # --- 阶段3：收敛反思 (串行处理) ---
    print("\n--- Stage 3: Refinement and Reflection ---")
    refine_prompt_formatted = refine_prompt.format(paper=paper, summary=summary_3)
    final_summary = call_ark_api(refine_prompt_formatted, temperature=0.0, top_p=1.0)
    if not final_summary:
        print("Failed to generate final refined summary.")
        return
    print("--- 最终反思摘要 ---\n", final_summary)

    return final_summary


if __name__ == "__main__":
    arxiv_id = "2309.04062"
    summary_func(arxiv_id)

