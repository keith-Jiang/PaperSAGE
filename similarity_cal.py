from sentence_transformers import SentenceTransformer
import time

models = {
    "bge-m3": "/home/zhangping/jrz-test/models/bge-m3",
    "all-MiniLM-L6-v2": "/home/zhangping/jrz-test/models/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "/home/zhangping/jrz-test/models/all-mpnet-base-v2"
}

sentence_1 = [
    "人工智能",
    "Large Language Model",
]

sentence_2 = [
    "Artificial Intelligence",
    "person",
    "检索增强生成",
    "good weather"
]

for model_name, model_path in models.items():
    print(model_name)
    model = SentenceTransformer(model_path)
    start_time = time.time()
    embeddings_1 = model.encode(sentence_1)
    embeddings_2 = model.encode(sentence_2)
    similarities = model.similarity(embeddings_1, embeddings_2)
    end_time = time.time()
    print(f"time: {end_time - start_time}")
    print(f"similarities: {similarities}")