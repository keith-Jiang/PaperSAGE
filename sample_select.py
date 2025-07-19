import json
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import random
import string

def read_json_file(file_path):
    """
    从 JSON 文件中读取数据
    :param file_path: JSON 文件路径
    :return: 数据列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data, file_path):
    """
    将数据保存到 JSON 文件中
    :param data: 数据列表
    :param file_path: JSON 文件路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def extract_keywords(sample):
    """
    提取样本中的关键词
    :param sample: 样本数据
    :return: 关键词字符串
    """
    data = json.loads(sample["output"])
    problem_keywords = data["keyword_problem"].replace(';', ' ')
    algorithm_keywords = data["keyword_algorithm"].replace(';', ' ')
    return problem_keywords + ' ' + algorithm_keywords

def add_random_chars(keyword):
    # 随机决定是否添加字符
    if random.random() < 0.9:
        # 随机生成 1 到 3 个字符
        random_chars = ''.join(random.choice(string.ascii_letters) for _ in range(random.randint(1, 10)))
        # 随机选择在开头、结尾或中间插入字符
        position = random.choice(['start', 'end', 'middle'])
        if position == 'start':
            keyword = random_chars + keyword
        elif position == 'end':
            keyword = keyword + random_chars
        else:
            index = random.randint(0, len(keyword))
            keyword = keyword[:index] + random_chars + keyword[index:]
    return keyword

def get_bert_embeddings(texts, model_name='BAAI/bge-large-en', save_path='embeddings.npy'):
    """
    使用 BGE-large 模型获取文本的嵌入向量，并保存结果
    :param texts: 文本列表
    :param model_name: 模型名称
    :param save_path: 保存 Embedding 结果的文件路径
    :return: 嵌入向量列表
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embeddings = []
    for text in tqdm(texts):
        inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**inputs)
            # BGE 模型的输出需要做平均池化得到最终的 embedding
            embedding = torch.mean(outputs.last_hidden_state, dim=1).numpy()
            embeddings.append(embedding.flatten())
    embeddings = np.array(embeddings)
    np.save(save_path, embeddings)
    return embeddings


def cluster_embeddings(embeddings, num_clusters=15000):
    """
    对嵌入向量进行聚类
    :param embeddings: 嵌入向量列表
    :param num_clusters: 聚类数量
    :return: 聚类标签
    """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels


def select_samples(data, labels, num_samples=15000):
    """
    根据聚类标签选择样本
    :param data: 数据列表
    :param labels: 聚类标签
    :param num_samples: 要选择的样本数量
    :return: 选择的样本列表
    """
    selected_samples = []
    for cluster_id in range(num_samples):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) > 0:
            # 选择每个聚类中的第一个样本
            selected_index = cluster_indices[0]
            selected_samples.append(data[selected_index])
    return selected_samples

input_file = "/home/zhangping/jrz-test/search_engine/paper_train.json"  
output_file = "output.json"  

# 读取数据
data = read_json_file(input_file) * 7
print(len(data))

# # 提取关键词
# keywords_list = [extract_keywords(sample) for sample in data]
# new_keywords_list = [keywords_list]

# for _ in range(6):
#     new_list = [add_random_chars(keyword) for keyword in keywords_list]
#     new_keywords_list.append(new_list)

# final_keywords_list = [keyword for sublist in new_keywords_list for keyword in sublist]
# print(len(final_keywords_list))
# print(final_keywords_list[-1])

# 获取 BERT 嵌入向量
# embeddings = get_bert_embeddings(final_keywords_list)

# 聚类
embeddings = np.load('embeddings.npy')
labels = cluster_embeddings(embeddings)
print("66666666666666666666666666")
# 选择样本
selected_samples = select_samples(data, labels)
# 保存数据
save_json_file(selected_samples, output_file)



    