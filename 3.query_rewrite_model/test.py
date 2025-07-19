import asyncio
import aiohttp
import time
import random
import json
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

# --- 依赖检查和导入 (transformers & torch) ---
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: 未找到 'transformers' 或 'torch' 库。本地直接模型推理功能将不可用。")
    print("请运行: pip install transformers torch sentencepiece accelerate") # accelerate 通常推荐

# --- 配置区 ---

# 场景1: 本地直接使用 transformers 运行模型 (无 vLLM)
# ----------------------------------------------------
# 你想在本地直接加载的 Hugging Face 模型标识符
# 例如 "Qwen/Qwen2-0.5B-Instruct", "Qwen/Qwen2-1.5B-Instruct" 等
LOCAL_HF_MODEL_ID = "/home/zhangping/jrz-test/models/sft/Qwen/Qwen2.5-0.5B" # 请确保与你的目标模型一致
# 可以选择使用的设备, "cuda" 代表GPU, "cpu" 代表CPU
LOCAL_MODEL_DEVICE = "cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu"

# 场景2: 本地通过 vLLM 运行模型 (有 vLLM)
# ----------------------------------------------------
# vLLM 服务 API 端点
VLLM_API_URL = "http://localhost:8003/v1/chat/completions" # 你的vLLM服务监听的聊天API端点
# vLLM 启动时加载并用于 API 请求的 'model' 字段值。
# 这个名称通常是你在启动 vLLM 时通过 --model 参数指定的模型的路径或 Hugging Face ID。
# 例如，如果 vLLM 启动命令是: python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2-0.5B-Instruct
# 那么 VLLM_EXPECTED_MODEL_NAME 就应该是 "Qwen/Qwen2-0.5B-Instruct" 或 vLLM 内部映射的名称。
VLLM_EXPECTED_MODEL_NAME = "Qwen2.5-0.5B" # 确保这与你vLLM服务加载的模型名匹配

# 全局模型和tokenizer实例 (用于本地直接推理)
# 将在 main 函数中根据需要加载
local_model = None
local_tokenizer = None

# --- 辅助函数 ---

def load_queries(filename: str) -> List[str]:
    """从文件加载查询，每行一个。"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        if not queries:
            print(f"警告: 查询文件 '{filename}' 为空或未找到有效查询。将使用默认查询。")
            return ["请帮我写一个关于北京旅游的攻略"]
        print(f"从 '{filename}' 加载了 {len(queries)} 条查询。")
        return queries
    except FileNotFoundError:
        print(f"错误: 查询文件 '{filename}' 未找到。将使用默认查询。")
        return ["请帮我写一个关于北京旅游的攻略"]

def prepare_vllm_request_payload(query: str, model_name_for_vllm: str) -> Dict[str, Any]:
    """准备 vLLM (OpenAI 兼容) API 的请求体。"""
    # 注意：此函数只返回 data payload，headers 在 run_load_test 中处理
    data = {
        "model": model_name_for_vllm,
        "messages": [
            {"role": "system", "content": "你是一个查询改写助手。"},
            {"role": "user", "content": f"请对以下查询进行改写，使其更清晰、更具体，便于搜索引擎或大模型理解。原始查询：\n'{query}'"}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
    }
    return data

# --- 核心请求/推理逻辑 ---

async def make_http_request(
    session: aiohttp.ClientSession,
    target_url: str,
    payload: Dict[str, Any], # payload 现在是 data
    query_for_log: str
) -> Tuple[Optional[float], bool, Optional[str]]:
    """发送单个异步 HTTP 请求 (用于 vLLM)。"""
    start_time = time.monotonic()
    headers = {'Content-Type': 'application/json'} # vLLM 通常不需要API Key
    try:
        timeout = aiohttp.ClientTimeout(total=60, connect=10) # 增加总超时
        async with session.post(target_url, headers=headers, json=payload, timeout=timeout) as response:
            response_text = await response.text()
            end_time = time.monotonic()
            latency = end_time - start_time
            if 200 <= response.status < 300:
                # 可以尝试解析json获取token数等信息，但这里简化处理
                return latency, True, None
            else:
                error_msg = f"HTTP Status: {response.status}, Query: {query_for_log[:50]}..., Response: {response_text[:200]}"
                return latency, False, error_msg
    except asyncio.TimeoutError:
        latency = time.monotonic() - start_time
        error_msg = f"Request Timeout, Query: {query_for_log[:50]}..."
        return latency, False, error_msg
    except aiohttp.ClientError as e:
        latency = time.monotonic() - start_time
        error_msg = f"Client Error: {e}, Query: {query_for_log[:50]}..."
        return latency, False, error_msg
    except Exception as e:
        latency = time.monotonic() - start_time
        error_msg = f"Unexpected HTTP Error: {type(e).__name__} - {e}, Query: {query_for_log[:50]}..."
        return latency, False, error_msg

def local_model_inference_sync(query: str, model: Any, tokenizer: Any, device: str) -> str:
    """
    使用加载的 Hugging Face 模型进行同步推理。
    此函数将在一个单独的线程中被调用以实现异步效果。
    """
    # 构建适用于模型的聊天模板输入
    messages = [
        {"role": "system", "content": "你是一个查询改写助手。"},
        {"role": "user", "content": f"请对以下查询进行改写，使其更清晰、更具体，便于搜索引擎或大模型理解。原始查询：\n'{query}'"}
    ]
    # 重要: Qwen2的聊天模板需要 apply_chat_template
    # 如果是旧版Qwen1.5可能需要手动拼接或使用其特定的tokenizer方法
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        # streamer = TextStreamer(tokenizer, skip_prompt=True) # 如果需要流式输出到控制台

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=100, # 与API的max_tokens对应
            # streamer=streamer, # 如果使用流式输出
            temperature=0.7, # 与API的temperature对应
            # top_p, top_k 等其他参数可以按需添加
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response_text # 返回生成的文本
    except Exception as e:
        print(f"本地模型推理错误: {e}")
        # raise # 将异常抛出，由外层捕获并记录
        return f"本地模型推理错误: {str(e)}"


async def make_local_model_request(
    query: str,
    model: Any, # AutoModelForCausalLM
    tokenizer: Any, # AutoTokenizer
    device: str
) -> Tuple[Optional[float], bool, Optional[str]]:
    """
    执行本地模型推理并计时。
    使用 asyncio.to_thread 将同步的推理函数包装为异步。
    """
    start_time = time.monotonic()
    error_msg: Optional[str] = None
    success = False
    try:
        # 将同步的推理函数放到线程池中执行
        _response_text = await asyncio.to_thread(local_model_inference_sync, query, model, tokenizer, device)
        # 这里可以添加对 _response_text 的检查，如果它包含错误信息则标记为失败
        if "本地模型推理错误:" in _response_text:
            success = False
            error_msg = _response_text
        else:
            success = True
    except Exception as e: # 捕获 local_model_inference_sync 中可能抛出的未处理异常
        error_msg = f"Local Inference Error: {type(e).__name__} - {e}, Query: {query[:50]}..."
        success = False

    latency = time.monotonic() - start_time
    return latency, success, error_msg


async def run_load_test(
    test_scenario: str, # "vllm" 或 "local_hf"
    test_name_prefix: str,
    queries: List[str],
    concurrency: int,
    duration: int,
    # vLLM specific
    vllm_target_url: Optional[str] = None,
    vllm_model_name_for_payload: Optional[str] = None,
    # Local HF model specific
    local_hf_model: Optional[Any] = None,
    local_hf_tokenizer: Optional[Any] = None,
    local_hf_device: Optional[str] = None,
    # Shared
    session: Optional[aiohttp.ClientSession] = None # 只有vLLM需要
) -> Dict[str, Any]:
    """运行指定并发数和时间的负载测试。"""

    test_full_name = f"{test_name_prefix} (并发 {concurrency})"
    print(f"\n--- 开始测试: {test_full_name} ---")
    if test_scenario == "vllm":
        print(f"场景: vLLM API ({vllm_model_name_for_payload})")
        print(f"目标 URL: {vllm_target_url}")
    elif test_scenario == "local_hf":
        print(f"场景: 本地 Transformers 模型 ({LOCAL_HF_MODEL_ID} on {local_hf_device})")
    else:
        raise ValueError(f"未知的测试场景: {test_scenario}")

    print(f"并发数: {concurrency}")
    print(f"持续时间: {duration} 秒")

    results: List[Tuple[Optional[float], bool, Optional[str]]] = []
    tasks = set()
    start_test_time = time.monotonic()
    total_requests_sent = 0

    # 注意：aiohttp.ClientSession 只在 vLLM 场景下创建和使用
    # 如果是 local_hf，session 参数应该为 None

    active_session = session # session from argument, only for vLLM

    try:
        while time.monotonic() - start_test_time < duration:
            if len(tasks) < concurrency:
                query = random.choice(queries)
                task = None
                if test_scenario == "vllm":
                    if not active_session or not vllm_target_url or not vllm_model_name_for_payload:
                        raise ValueError("vLLM 测试缺少必要的参数 (session, url, model_name)")
                    payload = prepare_vllm_request_payload(query, vllm_model_name_for_payload)
                    task = asyncio.create_task(make_http_request(active_session, vllm_target_url, payload, query))
                elif test_scenario == "local_hf":
                    if not local_hf_model or not local_hf_tokenizer or not local_hf_device:
                        raise ValueError("本地HF模型测试缺少必要的参数 (model, tokenizer, device)")
                    task = asyncio.create_task(make_local_model_request(query, local_hf_model, local_hf_tokenizer, local_hf_device))

                if task:
                    tasks.add(task)
                    total_requests_sent += 1
                    task.add_done_callback(tasks.discard)
            else: # len(tasks) == concurrency
                if tasks:
                    # 等待至少一个任务完成，为新任务腾出空间
                    _done, _pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                else: # 理论上不应发生，因为上面 len(tasks) < concurrency 会处理
                    await asyncio.sleep(0.001) # 安全暂停
            
            # 短暂休眠以避免CPU在任务队列满但没有任务完成时空转，并控制请求发起频率
            # 对于本地CPU/GPU绑定的推理，这个sleep可能影响不大，但对于HTTP请求有一定平滑作用
            await asyncio.sleep(0.001) # 可以根据实际情况调整或移除

        print(f"测试时间 ({duration}s) 到，等待剩余 {len(tasks)} 个任务完成...")
        if tasks:
            try:
                completed_results = await asyncio.gather(*tasks, return_exceptions=True)
                for res in completed_results:
                    if isinstance(res, Exception):
                        # gather 捕获的异常通常是任务内部未处理的异常
                        print(f"任务执行中捕获到未处理异常: {res}")
                        results.append((None, False, f"Unhandled task exception: {str(res)}"))
                    elif isinstance(res, tuple) and len(res) == 3:
                        results.append(res)
                    else:
                        print(f"收到未预期的任务结果: {res}")
                        results.append((None, False, f"Unexpected task result: {type(res)}"))
            except asyncio.CancelledError:
                print("等待任务被取消。")
            except Exception as e:
                 print(f"等待剩余任务完成时发生意外错误: {e}")
        print(f"所有任务处理完毕。共收集到 {len(results)} 条结果。")

    finally:
        # ClientSession 是在 main 中创建并传递的，这里不关闭
        pass


    end_test_time = time.monotonic()
    actual_duration = end_test_time - start_test_time

    latencies = [res[0] for res in results if res[0] is not None and res[1]]
    successful_requests = sum(1 for res in results if res[1])
    failed_requests = len(results) - successful_requests
    total_completed_requests = len(results) # 实际完成并收到某种结果（成功或失败信息）的请求

    rps = successful_requests / actual_duration if actual_duration > 0 else 0
    avg_latency = np.mean(latencies) * 1000 if latencies else float('nan') # ms
    p95_latency = np.percentile(latencies, 95) * 1000 if latencies else float('nan') # ms
    p99_latency = np.percentile(latencies, 99) * 1000 if latencies else float('nan') # ms
    min_latency = np.min(latencies) * 1000 if latencies else float('nan') # ms
    max_latency = np.max(latencies) * 1000 if latencies else float('nan') # ms
    success_rate = (successful_requests / total_completed_requests) * 100 if total_completed_requests > 0 else (100 if total_requests_sent == 0 else 0)

    print(f"--- 测试完成: {test_full_name} ---")
    print(f"实际测试时间: {actual_duration:.2f} 秒")
    print(f"总尝试发起请求/推理数 (估计): {total_requests_sent}")
    print(f"总完成请求/推理数 (收到响应/结果): {total_completed_requests}")
    print(f"成功数: {successful_requests}")
    print(f"失败数: {failed_requests}")
    print(f"成功率: {success_rate:.2f}%")
    print(f"吞吐量 (RPS/QPS): {rps:.2f}") # RPS for HTTP, QPS for local inference
    if latencies:
        print(f"平均延迟 (Avg Latency): {avg_latency:.2f} ms")
        print(f"最小延迟 (Min Latency): {min_latency:.2f} ms")
        print(f"最大延迟 (Max Latency): {max_latency:.2f} ms")
        print(f"P95 延迟: {p95_latency:.2f} ms")
        print(f"P99 延迟: {p99_latency:.2f} ms")
    else:
        print("无成功请求/推理，无法计算延迟统计。")

    errors = [res[2] for res in results if not res[1] and res[2]]
    if errors:
        print(f"部分错误信息样本 (最多5条):")
        for i, err in enumerate(errors[:min(5, len(errors))]):
            print(f"  - {str(err)[:300]}...") # 限制错误信息长度

    return {
        "test_name": test_full_name, # 使用包含并发信息的全名
        "scenario_type": test_scenario,
        "model_tested": LOCAL_HF_MODEL_ID if test_scenario == "local_hf" else vllm_model_name_for_payload,
        "concurrency": concurrency,
        "target_duration_s": duration,
        "actual_duration_s": actual_duration,
        "total_sent_approx": total_requests_sent,
        "total_completed": total_completed_requests,
        "successful": successful_requests,
        "failed": failed_requests,
        "success_rate_percent": success_rate,
        "rps_or_qps": rps,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency,
        "p99_latency_ms": p99_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "errors_sample": errors[:10]
    }


# --- 主程序入口 ---
async def main():
    global local_model, local_tokenizer # 声明使用全局变量

    parser = argparse.ArgumentParser(
        description="对比 本地直接运行Qwen模型(无vLLM) 和 本地通过vLLM运行Qwen模型(有vLLM) 的性能。"
    )
    parser.add_argument("--queries", type=str, default="queries.txt", help="包含测试查询的文件名 (每行一个)")
    parser.add_argument("--concurrency-levels", type=str, default="1,5,10", help="要测试的并发级别，用逗号分隔")
    parser.add_argument("--duration", type=int, default=20, help="每个并发级别的测试持续时间（秒）") # 增加默认时长以获得更稳定结果

    # 本地直接推理 (无vLLM) 相关参数
    parser.add_argument("--hf-model-id", type=str, default=LOCAL_HF_MODEL_ID,
                        help="用于本地直接推理的Hugging Face模型ID (例如 Qwen/Qwen2-0.5B-Instruct)")
    parser.add_argument("--hf-device", type=str, default=LOCAL_MODEL_DEVICE,
                        help="用于本地直接推理的设备 ('cuda' 或 'cpu')")
    parser.add_argument("--skip-local-hf-test", action="store_true", help="跳过本地直接模型推理测试")

    # 本地vLLM推理 相关参数
    parser.add_argument("--vllm-api-url", type=str, default=VLLM_API_URL, help="本地vLLM服务的API URL")
    parser.add_argument("--vllm-model-name", type=str, default=VLLM_EXPECTED_MODEL_NAME,
                        help="vLLM服务中目标模型的名称 (应与vLLM启动时加载的模型匹配)")
    parser.add_argument("--skip-vllm-test", action="store_true", help="跳过本地vLLM API测试")


    args = parser.parse_args()

    if not TRANSFORMERS_AVAILABLE and not args.skip_local_hf_test:
        print("错误: 'transformers' 或 'torch' 未安装，无法进行本地直接模型推理测试。")
        print("请安装它们或使用 --skip-local-hf-test 参数跳过此测试。")
        return

    queries = load_queries(args.queries)
    if not queries:
        print("错误: 无法加载查询，测试中止。")
        return

    concurrency_levels = [int(c) for c in args.concurrency_levels.split(',')]
    all_results = []

    # 场景1: 本地直接使用 transformers 运行模型 (无 vLLM)
    if not args.skip_local_hf_test and TRANSFORMERS_AVAILABLE:
        print(f"\n>>> 准备测试场景: 本地直接运行 {args.hf_model_id} (无vLLM) 在设备 {args.hf_device} 上 <<<")
        try:
            print(f"正在加载模型 '{args.hf_model_id}' 和 tokenizer 到设备 '{args.hf_device}'...")
            # 根据Qwen2的推荐，trust_remote_code=True可能是必要的
            # 对于0.5B模型，bf16/fp16可能在CPU上不支持，在GPU上可以尝试
            dtype_to_use = torch.bfloat16 if args.hf_device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
            if args.hf_device == "cpu": # CPU 上通常用 float32
                dtype_to_use = torch.float32
            
            local_tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id, trust_remote_code=True)
            local_model = AutoModelForCausalLM.from_pretrained(
                args.hf_model_id,
                torch_dtype=dtype_to_use, # 对于小模型，在GPU上用fp16/bf16，CPU上用fp32
                device_map=args.hf_device if args.hf_device == "cpu" else "auto", # "auto" 会用accelerate做多GPU或CPU+GPU
                trust_remote_code=True
            )
            if args.hf_device == "cuda" and hasattr(local_model, 'to'): # 确保模型在指定GPU设备
                 local_model.to(args.hf_device)
            local_model.eval() # 设置为评估模式

            print("模型和tokenizer加载完成。")

            for concurrency in concurrency_levels:
                hf_result = await run_load_test(
                    test_scenario="local_hf",
                    test_name_prefix=f"本地HF {args.hf_model_id.split('/')[-1]} (无vLLM)",
                    queries=queries,
                    concurrency=concurrency,
                    duration=args.duration,
                    local_hf_model=local_model,
                    local_hf_tokenizer=local_tokenizer,
                    local_hf_device=args.hf_device,
                    session=None # 本地推理不需要 aiohttp session
                )
                all_results.append(hf_result)
        except Exception as e:
            print(f"加载本地模型 {args.hf_model_id} 或执行测试时发生严重错误: {e}")
            if "out of memory" in str(e).lower():
                print("提示: 可能是显存不足。尝试使用更小的模型，或在CPU上运行。")
        finally:
            # 清理模型和tokenizer以释放内存/显存，特别是如果后续还有其他测试
            print("清理本地HF模型和tokenizer...")
            del local_model
            del local_tokenizer
            local_model = None
            local_tokenizer = None
            if args.hf_device == "cuda" and TRANSFORMERS_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("清理完成。")

    elif args.skip_local_hf_test:
        print("\n已跳过本地直接模型 (无vLLM) 测试。")
    elif not TRANSFORMERS_AVAILABLE:
         print("\n因缺少 'transformers'/'torch'，已跳过本地直接模型 (无vLLM) 测试。")


    # 场景2: 本地通过 vLLM 运行模型 (有 vLLM)
    if not args.skip_vllm_test:
        print(f"\n>>> 准备测试场景: 本地vLLM运行 {args.vllm_model_name} (有vLLM) <<<")
        print(f"vLLM API URL: {args.vllm_api_url}")
        print(f"请确保vLLM服务已在上述URL启动，并已加载模型 '{args.vllm_model_name}'。")
        
        # 为vLLM测试创建ClientSession
        # 调整连接池设置，使其与并发数相关
        # limit_per_host 应该至少是最大并发数
        max_concurrency_for_vllm = max(concurrency_levels) if concurrency_levels else 10
        conn = aiohttp.TCPConnector(
            limit_per_host=max_concurrency_for_vllm,  # 允许同时连接到同一主机的数量
            limit=max_concurrency_for_vllm * 2,       # 总连接池大小
            ssl=False # 本地vLLM通常是http
        )
        async with aiohttp.ClientSession(connector=conn) as vllm_session:
            for concurrency in concurrency_levels:
                vllm_result = await run_load_test(
                    test_scenario="vllm",
                    test_name_prefix=f"本地vLLM {args.vllm_model_name.split('/')[-1]} (有vLLM)",
                    queries=queries,
                    concurrency=concurrency,
                    duration=args.duration,
                    vllm_target_url=args.vllm_api_url,
                    vllm_model_name_for_payload=args.vllm_model_name,
                    session=vllm_session # 传递创建好的session
                )
                all_results.append(vllm_result)
    else:
        print("\n已跳过本地vLLM API测试。")


    if not all_results:
        print("\n没有执行任何测试。请检查命令行参数和依赖项。")
        return

    print("\n\n" + "="*20 + " 最终性能对比报告 " + "="*20)
    for result in all_results:
        print(f"\n--- {result['test_name']} ---")
        print(f"  测试场景: {'本地直接推理 (无vLLM)' if result['scenario_type'] == 'local_hf' else '本地vLLM API (有vLLM)'}")
        print(f"  测试模型: {result['model_tested']}")
        print(f"  并发数: {result['concurrency']}, 计划持续时间: {result['target_duration_s']}s (实际: {result['actual_duration_s']:.2f}s)")
        print(f"  吞吐量 (QPS/RPS): {result['rps_or_qps']:.2f}")
        print(f"  成功率: {result['success_rate_percent']:.2f}%")
        print(f"  平均延迟: {result['avg_latency_ms']:.2f} ms")
        print(f"  P95 延迟: {result['p95_latency_ms']:.2f} ms")
        print(f"  P99 延迟: {result['p99_latency_ms']:.2f} ms")
        print(f"  成功/失败数: {result['successful']}/{result['failed']}")
        if result['errors_sample'] and result['failed'] > 0:
            print(f"  部分错误样本:")
            for err_s in result['errors_sample'][:min(3, len(result['errors_sample']))]:
                 print(f"    - {str(err_s)[:200]}...")

if __name__ == "__main__":
    if not TRANSFORMERS_AVAILABLE:
        print("提示: 'transformers' 或 'torch' 库未完全导入，本地直接模型推理将不可用。")
        print("如果需要此功能，请确保已正确安装: pip install transformers torch sentencepiece accelerate")

    asyncio.run(main())
