import os
import re
import json
from openai import OpenAI
from transformers import AutoTokenizer
from prompts.ensemble_prompt import ensemble_prompt
from prompts.refine_prompt import refine_prompt
from prompts.summary_prompt import summary_prompt_v3
from config.config import config
from dotenv import load_dotenv
from config.config_param import config_param
from volcenginesdkarkruntime import Ark
import time

load_dotenv()

ARK_KEY = os.getenv("ARK_API_KEY")
BASE_URL = os.getenv("ARK_API_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3/")
TIMEOUT = int(os.getenv("ARK_TIMEOUT", 1800))
tokenizer_path = "/home/zhangping/jrz-test/models/sft/Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

with open("/home/zhangping/jrz-test/PaperSAGE/1.data_capture/transferred_papers/20250720/3d-denoisers-are-good-2d-teachers-molecular-pretraining-via-denoising-and-cross-modal-distillation_AAAI_2025.md", 'r', encoding='utf-8') as f:
    paper = f.read()

def generate_summary(prompt, paper, temperature, top_p):
    client = Ark(api_key=ARK_KEY, base_url=BASE_URL, timeout=TIMEOUT)
    start_time = time.time()
    summary_prompt = prompt.format(paper=paper)
    print(f"Prompt 长度: {len(tokenizer.encode(summary_prompt))}")
    completion = client.chat.completions.create(
        model="deepseek-v3-250324",
        messages=[
            {'role': 'user', 'content': summary_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        top_p=top_p
    )
    content = completion.choices[0].message.content
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"程序总运行时间: {total_time} 秒")

    return content

def collect_summary(prompt, paper, summary_1, summary_2, temperature, top_p):
    client = Ark(api_key=ARK_KEY, base_url=BASE_URL, timeout=TIMEOUT)
    start_time = time.time()
    summary_prompt = prompt.format(paper=paper, summary_1=summary_1, summary_2=summary_2)
    print(f"Prompt 长度: {len(tokenizer.encode(summary_prompt))}")
    completion = client.chat.completions.create(
        model="deepseek-v3-250324",
        messages=[
            {'role': 'user', 'content': summary_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        top_p=top_p
    )
    content = completion.choices[0].message.content
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"程序总运行时间: {total_time} 秒")

    return content

def refine_summary(prompt, paper, summary, temperature, top_p):
    client = Ark(api_key=ARK_KEY, base_url=BASE_URL, timeout=TIMEOUT)
    start_time = time.time()
    summary_prompt = prompt.format(paper=paper, summary=summary)
    print(f"Prompt 长度: {len(tokenizer.encode(summary_prompt))}")
    completion = client.chat.completions.create(
        model="deepseek-v3-250324",
        messages=[
            {'role': 'user', 'content': summary_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        top_p=top_p
    )
    content = completion.choices[0].message.content
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"程序总运行时间: {total_time} 秒")

    return content

if __name__ == "__main__":
    # 阶段1：发散探索
    summary_1 = generate_summary(summary_prompt_v3, paper, temperature=0.7, top_p=0.9)
    summary_2 = generate_summary(summary_prompt_v3, paper, temperature=0.7, top_p=0.9)
    print("--- 初始摘要 1 ---\n", summary_1)
    print("--- 初始摘要 2 ---\n", summary_2)

    # 阶段2：收敛整合
    summary_3 = collect_summary(ensemble_prompt, paper, summary_1, summary_2, temperature=0.0, top_p=1.0)
    print("--- 整合后摘要 ---\n", summary_3)

    # 阶段3：收敛反思
    final_summary = refine_summary(refine_prompt, paper, summary_3, temperature=0.0, top_p=1.0)
    print("--- 最终反思摘要 ---\n", final_summary)




    