import os
import json
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompts.ensemble_prompt import ensemble_prompt
from prompts.refine_prompt import refine_prompt
from prompts.summary_prompt import summary_prompt_v3
from dotenv import load_dotenv

# 加载vLLM模型
model_path = "/home/zhangping/jrz-test/models/sft/Qwen/Qwen2.5-7B"
llm = LLM(model=model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 读取论文
with open("/home/zhangping/jrz-test/PaperSAGE/1.data_capture/transferred_papers/20250720/3d-denoisers-are-good-2d-teachers-molecular-pretraining-via-denoising-and-cross-modal-distillation_AAAI_2025.md", 'r', encoding='utf-8') as f:
    paper = f.read()

def generate_with_vllm(prompt, temperature=0.7, top_p=0.9, max_tokens=2048):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    start_time = time.time()
    prompt_tokens = len(tokenizer.encode(prompt))
    print(f"Prompt长度: {prompt_tokens} tokens")
    
    outputs = llm.generate([prompt], sampling_params)
    content = outputs[0].outputs[0].text
    
    end_time = time.time()
    print(f"生成耗时: {end_time - start_time:.2f}秒")
    
    return content

def generate_summary(prompt, paper, temperature, top_p):
    formatted_prompt = prompt.format(paper=paper)
    return generate_with_vllm(formatted_prompt, temperature, top_p)

def collect_summary(prompt, paper, summary_1, summary_2, temperature, top_p):
    formatted_prompt = prompt.format(paper=paper, summary_1=summary_1, summary_2=summary_2)
    return generate_with_vllm(formatted_prompt, temperature, top_p)

def refine_summary(prompt, paper, summary, temperature, top_p):
    formatted_prompt = prompt.format(paper=paper, summary=summary)
    return generate_with_vllm(formatted_prompt, temperature, top_p)

if __name__ == "__main__":
    # 阶段1：发散探索
    print("生成初始摘要1...")
    summary_1 = generate_summary(summary_prompt_v3, paper, temperature=0.7, top_p=0.9)
    # print("生成初始摘要2...")
    # summary_2 = generate_summary(summary_prompt_v3, paper, temperature=0.7, top_p=0.9)
    
    print("\n--- 初始摘要 1 ---\n", json.loads(summary_1))
    # print("\n--- 初始摘要 2 ---\n", json.loads(summary_2))

    # # 阶段2：收敛整合
    # print("\n整合摘要中...")
    # summary_3 = collect_summary(ensemble_prompt, paper, summary_1, summary_2, temperature=0.0, top_p=1.0)
    # print("\n--- 整合后摘要 ---\n", json.loads(summary_3))

    # # 阶段3：收敛反思
    # print("\n优化摘要中...")
    # final_summary = refine_summary(refine_prompt, paper, summary_3, temperature=0.0, top_p=1.0)
    # print("\n--- 最终摘要 ---\n", json.loads(final_summary))
