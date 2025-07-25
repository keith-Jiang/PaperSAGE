import json
import streamlit as st
from json import JSONDecoder

with open("/home/zhangping/jrz-test/PaperSAGE/1.data_capture/database/extra_1/2408.09918.json", "r", encoding='utf-8') as f:
    paper = json.load(f)

json_str = paper["summary"].strip().split("\n", 1)[1].rsplit("\n", 1)[0]
# summary = json.loads(json_str)
decoder = JSONDecoder()
try:
    summary = decoder.decode(json_str)
except json.JSONDecodeError as e:
    print(f"Error at position {e.pos}: {e.doc[e.pos-10:e.pos+10]}")

# 3. 现在 summary 是一个字典，可以正常访问
st.markdown(summary["core_summary"])
st.markdown(summary["algorithm_details"])
st.markdown(summary["comparative_analysis"])
st.markdown(summary["keywords"])
