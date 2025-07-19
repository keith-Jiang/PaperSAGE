import os
import re
import json
from openai import OpenAI
from prompts.summary_prompt import summary_prompt_v1, summary_prompt_v2, summary_prompt_v3
from config.config import config
from dotenv import load_dotenv
from config.config_param import config_param
from volcenginesdkarkruntime import Ark
import time

load_dotenv()
DEFAULT_MODEL = os.getenv("ARK_MODEL", "deepseek-v3-250324")
with open("/home/zhangping/jrz-test/PaperSAGE/1.data_capture/transferred_papers/20250720/3d-denoisers-are-good-2d-teachers-molecular-pretraining-via-denoising-and-cross-modal-distillation_AAAI_2025.md", 'r', encoding='utf-8') as f:
    paper = f.read()

def generate_summary(prompt, paper, temper, topp, write_to_json=False, json_file_path="result.json"):
    ark_key = os.getenv("ARK_API_KEY")
    base_url = os.getenv("ARK_API_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3/")
    timeout=int(os.getenv("ARK_TIMEOUT", 1800))
    client = Ark(api_key=ark_key, base_url=base_url, timeout=timeout)

    start_time = time.time()
    summary_prompt = prompt.format(paper=paper)
    completion = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {'role': 'user', 'content': summary_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=temper,
        top_p=topp
    )
    content = completion.choices[0].message.content
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"程序总运行时间: {total_time} 秒")

    if write_to_json:
        result = {
            "content": content
        }
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
        print(f"结果已写入 {json_file_path}")
    return content

if __name__ == "__main__":
    content = generate_summary(summary_prompt_v3, paper,
                              config_param["en_summary_temper"], config_param["en_summary_topp"], write_to_json=False)
    print(content)
    