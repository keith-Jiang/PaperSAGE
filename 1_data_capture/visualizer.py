import json
import streamlit as st
from pathlib import Path

# 设置页面配置
st.set_page_config(layout="wide", page_title="JSON文件浏览器")

# 初始化session state
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'json_files' not in st.session_state:
    st.session_state.json_files = []

def load_json_files(folder_path):
    """加载指定文件夹中的所有JSON文件"""
    try:
        path = Path(folder_path)
        if not path.exists():
            return None, "文件夹不存在"
        if not path.is_dir():
            return None, "路径不是文件夹"
        
        json_files = sorted(list(path.glob("*.json")))
        if not json_files:
            return None, "文件夹中没有JSON文件"
        
        return json_files, None
    except Exception as e:
        return None, f"加载文件时出错: {str(e)}"

def display_json_content(file_path):
    """显示单个JSON文件的内容"""
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            paper = json.load(f)
        
        # 解析summary字段
        json_str = paper["summary"].strip().split("\n", 1)[1].rsplit("\n", 1)[0]
        summary = json.loads(json_str)
        
        # 创建选项卡
        tab1, tab2, tab3, tab4 = st.tabs(["核心概要", "算法细节", "对比分析", "关键词"])
        
        with tab1:
            st.subheader("核心概要")
            st.markdown(summary["core_summary"])
        
        with tab2:
            st.subheader("算法细节")
            st.markdown(summary["algorithm_details"])
        
        with tab3:
            st.subheader("对比分析")
            st.markdown(summary["comparative_analysis"])
        
        with tab4:
            st.subheader("关键词")
            st.markdown(summary["keywords"])
            
    except Exception as e:
        st.error(f"解析文件时出错: {str(e)}")

# 主界面
st.title("JSON文件浏览器")

# 文件夹输入
folder_path = st.text_input(
    "输入包含JSON文件的文件夹路径:",
    value="/home/zhangping/jrz-test/PaperSAGE/1.data_capture/database/20250721/"
)

if st.button("加载文件夹"):
    json_files, error = load_json_files(folder_path)
    if error:
        st.error(error)
    else:
        st.session_state.json_files = json_files
        st.session_state.current_index = 0
        st.success(f"找到 {len(json_files)} 个JSON文件")

# 如果有加载的文件，显示导航和内容
if st.session_state.json_files:
    st.divider()
    
    # 导航控制
    col1, col2, col3 = st.columns([1, 4, 1])
    
    with col1:
        if st.button("⬅️ 上一个", disabled=st.session_state.current_index == 0):
            st.session_state.current_index -= 1
    
    with col3:
        if st.button("下一个 ➡️", disabled=st.session_state.current_index == len(st.session_state.json_files)-1):
            st.session_state.current_index += 1
    
    with col2:
        current_file = st.session_state.json_files[st.session_state.current_index]
        st.info(f"正在显示: {current_file.name} ({st.session_state.current_index + 1}/{len(st.session_state.json_files)})")
    
    # 显示当前文件内容
    display_json_content(current_file)
else:
    st.info("请输入文件夹路径并点击'加载文件夹'按钮")
