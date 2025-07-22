import json
import streamlit as st

with open("/home/zhangping/jrz-test/PaperSAGE/1.data_capture/database/20250720/2312.03701.json", "r", encoding='utf-8') as f:
    paper = json.load(f)

# 1. 检查 paper["summary"] 是否是字符串
if isinstance(paper["summary"], str):
    # 2. 解析它（因为它是一个 JSON 字符串）
    print(paper["summary"][1250: 1300])
    text = paper["summary"].replace("\\", "\\\\") 
    summary = json.loads(text)
else:
    # 如果 paper["summary"] 已经是字典，直接使用
    summary = paper["summary"]

# 3. 现在 summary 是一个字典，可以正常访问
st.markdown(summary["core_summary"].replace("\\n", "  \n"))
st.markdown(summary["algorithm_details"].replace("\\n", "  \n"))
st.markdown(summary["comparative_analysis"].replace("\\n", "  \n"))
st.markdown(summary["keywords"].replace("\\n", "  \n"))
