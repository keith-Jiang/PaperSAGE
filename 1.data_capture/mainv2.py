from config.config import config
from config.config_param import config_param
from utils.query_qwen import QueryTongyi
from utils.utils import extract_json_content, parse_refine_res, parse_summary_res, get_tongyi_result
from prompts.summary_prompt import summary_prompt_final
from prompts.ensemble_prompt import ensemble_prompt
from prompts.refine_prompt import refine_prompt
from prompts.modify_prompt import modify_prompt
import time
from data.data import data

def ensemble(queryGen, inputs, config_param):
    summary_prompt_str = summary_prompt_final.format(paper=inputs)
    en_summary_res = []
    i = 6
    while len(en_summary_res) <= 3 and i >= 0:
        i -= 1
        en_summary_res_tmp = get_tongyi_result(queryGen, summary_prompt_str, config_param['en_summary_temp'], config_param['en_summary_topp'])
        en_summary_res_tmp_str = parse_summary_res(en_summary_res_tmp)
        if en_summary_res_tmp_str is None:
            continue
        print(en_summary_res_tmp_str)
        en_summary_res.append(en_summary_res_tmp_str)   
         
    ensemble_prompt_str = ensemble_prompt.format(paper=inputs, summary1=en_summary_res[0], summary2=en_summary_res[1], summary3=en_summary_res[2])

    i = 3
    while len(res)<=1 and i>=0:
        i -= 1
        ensemble_summary_res = get_tongyi_result(queryGen, ensemble_prompt_str, config_param['en_summary_all_temp'], config_param['en_summary_all_topp'])
        res_tmp = parse_summary_res(ensemble_prompt_str)
        if res_tmp is None:
            continue 
        print(res_tmp)
        return res_tmp
    return None

def refine(queryGen, paper, init_summary, config_param, max_iter=3):
    for ii in range(max_iter):
        refine_prompt_str = refine_prompt.format(paper=paper, summary=init_summary)
        i = 3
        sug = None
        sug_res = None
        while sug is None and i >= 0:
            i -= 1
            refine_summary_res = get_tongyi_result(queryGen, refine_prompt_str, config_param['re_summary_all_temp'], config_param['re_summary_all_topp'])
            sug_tmp, sug_res_tmp = parse_refine_res(refine_summary_res)
            if sug_tmp is None:
                continue 
            sug = sug_tmp
            sug_res = sug_res_tmp
            break
        print(sug)
        if sug is None:
            return None 
        if sug_res == '无需修改':
            return init_summary
        modify_prompt_str = modify_prompt.format(paper=paper, summary=init_summary, suggestion=sug)
        i = 3
        modify_res = None
        while modify_res is None and i>=0:
            i -= 1
            modify_res_str = get_tongyi_result(queryGen, modify_prompt_str, config_param['mo_summary_all_temp'], config_param['mo_summary_all_topp'])
            modify_res_tmp = parse_summary_res(modify_res_str)
            if modify_res_tmp is None:
                continue 
            modify_res = modify_res_tmp
            break
        print(modify_res)
        if modify_res is not None:
            init_summary = modify_res
    return init_summary

if __name__ == '__main__':
    queryGen = QueryTongyi(config.get('api_key'))
    ensemble_res = ensemble(queryGen, data, config_param)
    res = refine(queryGen, data, ensemble_res, config_param)