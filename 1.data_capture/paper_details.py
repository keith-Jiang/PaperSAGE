import streamlit as st
import json
import re
from pathlib import Path
import sys

# 将主目录添加到sys.path以导入app.py中的函数
# 这是一个在多页面应用中共享函数的常用技巧
sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from app import translate_text_api, rag_answer_question
except ImportError:
    st.error("无法从主应用导入API函数。")
    st.stop()


st.set_page_config(page_title="论文详情", layout="wide")

# --- 0. 检查和加载数据 ---
if 'selected_paper_path' not in st.session_state:
    st.error("请先在主页选择一篇论文。")
    st.page_link("app.py", label="返回主页", icon="🏠")
    st.stop()

try:
    paper_path = Path(st.session_state['selected_paper_path'])
    with open(paper_path, 'r', encoding='utf-8') as f:
        paper_data = json.load(f)
except Exception as e:
    st.error(f"加载论文数据失败: {e}")
    st.page_link("app.py", label="返回主页", icon="🏠")
    st.stop()

# --- 1. 页面标题和元数据 ---
st.page_link("app.py", label="返回主页", icon="🏠")
st.title(paper_data.get('title', '未知标题'))

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"**✍️ 作者:** {', '.join(paper_data.get('authors', []))}")
with col2:
    st.markdown(f"**🗓️ 发表日期:** {paper_data.get('publication_date', 'N/A')}")
with col3:
    st.link_button("🔗 ArXiv 页面", paper_data.get('link', '#'))
    st.link_button("📄 PDF 下载", paper_data.get('pdf_link', '#'))

st.divider()


# --- 2. RAG 问答区域 ---
st.header("❓ 对论文提问 (RAG)")
with st.container(border=True):
    user_question = st.text_input("输入你关于这篇论文的问题:", placeholder="例如：这篇论文使用了什么模型？")
    if st.button("提交问题"):
        if user_question:
            with st.spinner("正在思考答案..."):
                answer = rag_answer_question(user_question, paper_data.get('paper_content', ''))
                st.info(answer)
        else:
            st.warning("请输入一个问题。")

st.divider()

# --- 3. 论文原文与逐段翻译 ---
st.header("📖 论文原文 & 逐段翻译")

paper_content = paper_data.get('paper_content', '无原文内容。')

# 使用正则表达式按 Markdown 标题分割文章
# re.split 会保留分隔符（标题），所以我们需要将它们配对起来
sections = re.split(r'(\n#+ .*\n)', paper_content)

# 第一个元素通常是标题前的空字符串或引言，单独处理
if sections[0].strip():
    st.markdown(sections[0])
    # 为没有标题的引言部分也添加翻译按钮
    if st.button("翻译引言部分", key="translate_intro"):
         with st.spinner("正在翻译..."):
            translation = translate_text_api(sections[0])
            st.success("翻译结果:")
            st.markdown(translation)

# 循环处理成对的 "标题" 和 "内容"
# 从索引1开始，步长为2
for i in range(1, len(sections), 2):
    header = sections[i].strip()
    content = sections[i+1].strip()
    
    with st.container(border=True):
        st.subheader(header)
        st.markdown(content)
        
        # 使用唯一的key来避免Streamlit的 "DuplicateWidgetID" 错误
        button_key = f"translate_{paper_path.stem}_{i}"
        
        if st.button("翻译此段", key=button_key):
            with st.spinner("正在翻译..."):
                translation = translate_text_api(content)
                st.success("翻译结果:")
                st.markdown(translation)