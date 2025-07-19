import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import os
import time
from tqdm import tqdm

# --- 配置 ---
JSON_FILE_PATH = '/home/zhangping/jrz-test/search_engine/rag_data/data2.json'
MODEL_NAME = '/home/zhangping/jrz-test/models/bge-m3'
K_NEAREST_NEIGHBORS = 3

# --- 1. 加载数据 ---
def load_json_data(json_path):
    outputs = []
    original_queries = []
    semantic_queries = []

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for i in tqdm(range(len(data))):
            outputs.append(data[i]["output"])
            original_queries.append(data[i]["query"])
            output = json.loads(data[i]["output"].replace("```json", "").replace("```", "").strip(), strict=False)
            query = output.get("语义query")
            semantic_queries.append(query)

    return outputs, original_queries, semantic_queries

# --- 2. 文本 Embedding ---
def get_embeddings(texts, model_name):
    print(f"正在加载 embedding 模型: {model_name}...")
    start_time = time.time()
    # 可以在这里指定 device='cuda' 如果有 GPU 并且安装了 PyTorch 的 CUDA 版本
    model = SentenceTransformer(model_name, device='cuda') # 改为 'cuda' 使用 GPU
    print(f"模型加载完成，耗时 {time.time() - start_time:.2f} 秒。")

    print(f"开始进行 embedding，共 {len(texts)} 条文本...")
    start_time = time.time()
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=128) # 调整 batch_size
    print(f"Embedding 完成，耗时 {time.time() - start_time:.2f} 秒。")
    return embeddings

# --- 3. 构建 Faiss 索引 ---
def build_faiss_index(embeddings):
    """构建 Faiss 索引 (使用 IndexFlatL2)"""
    if embeddings is None or len(embeddings) == 0:
        print("错误: 没有有效的 embeddings 来构建索引。")
        return None

    dimension = embeddings.shape[1]  # 获取 embedding 的维度
    print(f"Embedding 维度: {dimension}")

    # 标准化 embeddings (对于 L2 索引，这使得 L2 距离搜索等价于 cosine 相似度搜索)
    print("正在标准化 embeddings...")
    faiss.normalize_L2(embeddings)

    print("正在构建 Faiss 索引 (IndexFlatL2)...")
    start_time = time.time()
    # IndexFlatL2: 简单的暴力 L2 距离搜索，适合中小型数据集，精度最高
    # 更多索引类型请参考 Faiss 文档和下面的优化技巧部分
    index = faiss.IndexFlatL2(dimension)

    # 将 embeddings 添加到索引
    index.add(embeddings)
    print(f"Faiss 索引构建完成，共添加 {index.ntotal} 个向量，耗时 {time.time() - start_time:.2f} 秒。")
    return index

# --- 4. 相似性搜索 ---
def search_similar(query_text, model, index, original_data, original_queries, k):
    """
    在 Faiss 索引中搜索与查询文本最相似的条目。
    返回: 最相似的 k 个原始条目及其距离。
    """
    if index is None:
        print("错误: Faiss 索引未初始化。")
        return []
    if not query_text:
        print("错误: 查询文本不能为空。")
        return []

    print(f"\n正在搜索与查询 '{query_text}' 最相似的 {k} 条记录...")
    start_time = time.time()

    # 1. 对查询文本进行 embedding
    query_embedding = model.encode([query_text])

    # 2. 标准化查询 embedding (与索引中的向量保持一致)
    faiss.normalize_L2(query_embedding)

    # 3. 执行搜索
    # 返回的 distances 是 L2 距离的平方。由于向量已标准化，
    # D^2 = ||v1 - v2||^2 = (v1-v2).(v1-v2) = v1.v1 - 2*v1.v2 + v2.v2
    #      = 1 - 2*cos(theta) + 1 = 2 - 2*cos(theta)
    # 距离越小，cos(theta) 越大，即相似度越高。
    distances, indices = index.search(query_embedding, k)

    search_time = time.time() - start_time
    print(f"搜索完成，耗时 {search_time:.4f} 秒。")

    # 4. 获取结果
    results = []
    if indices.size > 0:
        for i in range(len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            if idx < len(original_data): # 确保索引有效
                 # 查找对应的原始数据
                original_entry = original_data[idx]
                original_query = original_queries[idx]
            
                results.append({
                    "similarity_score": 1 - dist / 2, # 将L2距离平方转换为余弦相似度 (近似)
                    "distance_l2_sq": dist,
                    "original_query": original_query,
                    "original_entry": original_entry
                })
            else:
                print(f"警告：搜索返回的索引 {idx} 超出原始数据范围 {len(original_data)}")
    return results

# --- 主程序 ---
if __name__ == "__main__":
    # 1. 加载和预处理数据
    print("--- 步骤 1: 加载和预处理数据 ---")
    outputs, original_queries, semantic_queries = load_json_data(JSON_FILE_PATH)
    
    if not outputs:
        print("未能加载或找到有效数据，程序退出。")
        exit()

    # 2. 获取 Embeddings
    print("\n--- 步骤 2: 获取 Embeddings ---")
    embedding_model = SentenceTransformer(MODEL_NAME, device='cpu') # 加载模型一次
    embeddings_np = get_embeddings(semantic_queries, MODEL_NAME) # 使用已加载的模型

    # 3. 构建 Faiss 索引
    print("\n--- 步骤 3: 构建 Faiss 索引 ---")
    faiss_index = build_faiss_index(embeddings_np)

    # 4. 搜索示例
    if faiss_index:
        print("\n--- 步骤 4: 搜索示例 ---")
        # 示例查询
        # query = "有关大型语言模型在医疗领域的应用论文"
        query = "卷积神经网络用于图像识别的最新进展"
        # query = "transformer 在自然语言处理中的作用"
        # query = "脑肿瘤分类方法" # 这个应该能匹配到示例数据

        search_results = search_similar(query, embedding_model, faiss_index, outputs, original_queries, K_NEAREST_NEIGHBORS)

        print(f"\n--- 查询 '{query}' 的 Top {K_NEAREST_NEIGHBORS} 相似结果 ---")
        if search_results:
            for i, result in enumerate(search_results):
                print(f"\n--- 结果 {i+1} ---")
                print(f"相似度分数 (Cosine Similarity): {result['similarity_score']:.4f}")
                # print(f"L2 距离平方: {result['distance_l2_sq']:.4f}")
                print(f"召回的语义 Query: {result['original_query']}")
                print(f"原始完整条目:")
                print(json.dumps(result['original_entry'], indent=2, ensure_ascii=False)) # 打印格式化的原始 JSON
        else:
            print("没有找到相似的结果。")

    else:
        print("Faiss 索引构建失败，无法进行搜索。")

