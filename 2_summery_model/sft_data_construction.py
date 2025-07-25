from prompts.summary_prompt import summary_prompt_v1, summary_prompt_v2, summary_prompt_v3

results = []
for paper in papers:
    summary_prompt = summary_prompt_v3.format(paper=paper)
    results.append({"instruction": summary_prompt, "input": "", "output": ""})

with open("/home/zhangping/jrz-test/search_engine/2.summery_model/total_data.json", "w", encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(results[0])