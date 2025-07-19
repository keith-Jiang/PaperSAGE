import json
import time

def extract_json_content(content):
    # 找到json字符串的位置，返回解析后的json对象
    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1:
        json_str = content[start:end+1]
        try:
            json_ss = json.loads(json_str)
            return json_ss
        except:
            return None
    return None

def parse_summary_res(inputs):
    ct = extract_json_content(inputs)
    res = None
    try:
        if ct is not None:
            res="论文总结："+ct['summary']+"\n算法简介："+ct['algorithm']+'\n对比结果：'+ct['compare_result']+'\n业务关键词：'+ct['keyword_problem']+'\n算法关键词：'+ct['keyword_algorithm']
    except Exception as e:
        print(str(e))
    return res

def parse_refine_res(inputs):
    res = extract_json_content(inputs)
    ct = None
    ct_res = None
    try:
        if res is not None:
            ct = "分析："+res['分析']+'\n修改建议：'+res['修改建议']
            ct_res = res['结果']
    except Exception as e:
        print(str(e))
    return ct, ct_res

def get_tongyi_result(queryGen, new_prompt, temperature, top_p):
    res = None
    try:
        res = queryGen.query(new_prompt, temperature=temperature, top_p=top_p)
    except Exception as e:
        tmp = 0
        while tmp <= 10:
            res = queryGen.query(new_prompt, temperature=temperature, top_p=top_p)
            if res is not None and res["status"]:
                break
            print(f'通义接口调用失败，正在第{tmp}次重试.....error:{e}')
            tmp += 1
    i = 0
    while (not res['status'] or len(res['content']) < 5) and i < 10:
        res = queryGen.query(new_prompt, temperature=temperature, top_p=top_p)
        i += 1
        time.sleep(10)
        print(f'通义接口调用失败，间隔10s，正在第{i}次重试.....')

    return res.get('content',None)

