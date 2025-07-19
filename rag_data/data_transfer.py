import json
import re
from tqdm import tqdm

new_data = []

with open("/home/zhangping/jrz-test/search_engine/rewrite_query_train.jsonl", "r", encoding="utf-8") as f:
    data = json.load(f)

    for i in tqdm(range(len(data))):
        olriginal_query = data[i]["instruction"].split("\n")[-2].split(":")[-1]
        output = data[i].get('output', '')
        res = json.loads((output.replace("```json", "").replace("```", "").strip()))
        semantic_query = res.get("语义query")

        entry = {
            "original_query": olriginal_query,
            "output": output,
            "semantic_query": semantic_query
        }

        new_data.append(entry)

with open('/home/zhangping/jrz-test/search_engine/rag_data/data2.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)

