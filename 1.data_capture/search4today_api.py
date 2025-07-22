import streamlit as st
import os
import re
import glob
import json
import random
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import concurrent.futures
# from transformers import AutoTokenizer # 暂时用不到
# from bs4 import BeautifulSoup # 暂时用不到

# =================================================================
# 0. 模块导入与占位符
# =================================================================
# --- 假设的外部模块 ---
# 我们先用占位符函数替代，以确保UI可以独立运行
def summary_func(arxiv_id):
    """占位符：模拟调用大模型生成总结"""
    st.toast(f"正在为 {arxiv_id} 生成总结...")
    time.sleep(2) # 模拟耗时
    # 模拟返回一个包含Markdown的JSON字符串
    summary_data = {
        "一句话总结": "### 🚀 一句话总结\n\n这是一篇关于 Transformer 模型的开创性论文，提出了'Attention Is All You Need'的核心思想。",
        "核心贡献": "### 🎯 核心贡献\n\n- **自注意力机制**: 完全替代了循环和卷积结构。\n- **位置编码**: 解决了序列中词语的位置信息问题。\n- **多头注意力**: 允许模型在不同表示子空间中共同关注来自不同位置的信息。"
    }
    return json.dumps(summary_data, ensure_ascii=False, indent=4)

def translate_text(text_to_translate):
    """占位符：模拟调用翻译API"""
    st.toast(f"正在翻译选定文本...")
    time.sleep(1) # 模拟耗时
    return f"【翻译结果】\n\n{text_to_translate}"

def rag_query(query, paper_content):
    """占位符：模拟调用RAG进行问答"""
    st.toast(f"正在基于原文内容回答问题: {query}")
    time.sleep(2) # 模拟耗时
    return f"关于您的问题 “{query}”，根据论文内容，答案是... (这是一个模拟的RAG回答)。"


# =================================================================
# 1. 全局配置和初始化
# =================================================================
st.set_page_config(page_title="PaperSAGE - ArXiv论文助手", layout="wide")

# --- 路径配置 ---
BASE_PATH = Path("/home/zhangping/jrz-test/PaperSAGE/1.data_capture") 
DB_ROOT_PATH = BASE_PATH / "database"

# --- API 和模型配置 ---
# 警告：请勿将敏感信息硬编码。建议使用st.secrets或环境变量。
API_TOKEN = 'Bearer eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI2MjQwMzE5MCIsInJvbCI6IlJPTEVfUkVHSVNURVIiLCJpc3MiOiJPcGVuWExhYiIsImlhdCI6MTc1Mjg0Mzc4NCwiY2xpZW50SWQiOiJsa3pkeDU3bnZ5MjJqa3BxOXgydyIsInBob25lIjoiIiwib3BlbklkIjpudWxsLCJ1dWlkIjoiMzhkZDYzMjAtNzQ3Ny00ZjhjLTgwNTYtMWE0NjliNWUyZDc4IiwiZW1haWwiOiIiLCJleHAiOjE3NTQwNTMzODR9.2MBgX_jgoq6Z6XZLcMoi7YLZuJdQ_Yb2GXRh8SA0KglP0LWQZjUOTsv-xpgrIjCNGf9nBrsyRtg9CmUjIt6g0Q'
BASE_URL = 'https://mineru.net/api/v4'
POLLING_INTERVAL = 10

# --- ArXiv CS 分类 ---
ARXIV_CATEGORIES = {
    "计算机科学 (CS)": [
        "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "cs.RO",
        "cs.CR", "cs.DS", "cs.IT", "cs.SE"
    ]
    # 可以添加更多顶级分类
}


# =================================================================
# 2. 后端功能函数 (与之前版本基本相同)
# =================================================================

# --- 步骤 1: 下载PDF ---
def download_arxiv_pdf(arxiv_url, save_dir):
    """从ArXiv URL下载PDF文件。"""
    try:
        if not re.match(r'https?://arxiv\.org/abs/\d+\.\d+', arxiv_url):
            return None, "URL格式不正确，应为 'https://arxiv.org/abs/...' 格式。"
        
        pdf_url = arxiv_url.replace('/abs/', '/pdf/')
        arxiv_id = arxiv_url.split('/')[-1]
        pdf_filename = f"{arxiv_id}.pdf"
        save_path = save_dir / pdf_filename

        response = requests.get(pdf_url, stream=True, timeout=60)
        response.raise_for_status()

        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return save_path, None
    except requests.exceptions.RequestException as e:
        return None, f"下载PDF失败: {e}"
    except Exception as e:
        return None, f"发生未知错误: {e}"

# --- 步骤 2: PDF转Markdown (单文件版本) ---
def convert_single_pdf_to_md(pdf_path, output_folder):
    """使用API将单个PDF转换为Markdown。"""
    api_headers = {'Content-Type': 'application/json', 'Authorization': API_TOKEN}
    filename = pdf_path.name
    
    try:
        response = requests.post(f'{BASE_URL}/file-urls/batch', headers=api_headers, json={"files": [{"name": filename}]})
        response.raise_for_status()
        result = response.json()
        if result.get("code") != 0: return None, f"获取上传URL失败: {result.get('msg')}"
        
        batch_id = result["data"]["batch_id"]
        upload_url = result["data"]["file_urls"][0]

        with open(pdf_path, 'rb') as f:
            res_upload = requests.put(upload_url, data=f)
            if res_upload.status_code != 200: return None, f"上传失败, 状态码: {res_upload.status_code}"

        query_url = f"{BASE_URL}/extract-results/batch/{batch_id}"
        start_time = time.time()
        while time.time() - start_time < 300: # 5分钟超时
            res_query = requests.get(query_url, headers=api_headers)
            if res_query.status_code == 200:
                result = res_query.json()
                if result.get("code") == 0 and result.get("data", {}).get("extract_result"):
                    task = result["data"]["extract_result"][0]
                    if task.get("state") == "done":
                        zip_url = task.get("full_zip_url")
                        zip_response = requests.get(zip_url, timeout=180)
                        zip_response.raise_for_status()
                        
                        with ZipFile(BytesIO(zip_response.content)) as z:
                            md_name_in_zip = next((item for item in z.namelist() if item.lower().endswith('.md')), None)
                            if md_name_in_zip:
                                md_content = z.read(md_name_in_zip).decode('utf-8')
                                md_path = output_folder / f"{pdf_path.stem}.md"
                                with open(md_path, 'w', encoding='utf-8') as f_out:
                                    f_out.write(md_content)
                                return md_path, None
                            else:
                                return None, "结果压缩包中未找到.md文件。"
                    elif task.get("state") == "failed":
                        return None, f"API处理失败: {task.get('error_msg', '未知错误')}"
            time.sleep(POLLING_INTERVAL)
        return None, "轮询结果超时。"
    except Exception as e:
        return None, f"PDF到MD转换过程中出错: {e}"

# --- 步骤 3: 清理Markdown文件 (单文件版本) ---
def process_single_md_file(md_path, pdf_to_delete_path):
    """清理单个MD文件并删除对应的PDF。"""
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            paper_content = f.read()

        sections_to_remove = ["Acknowledgement", "Acknowledgements", "Reference", "References", "Acknowledgment", "Acknowledgments",
                              "ACKNOWLEDGEMENT", "ACKNOWLEDGEMENTS", "REFERENCE", "REFERENCES", "ACKNOWLEDGMENT", "ACKNOWLEDGMENTS"]
        
        cut_off_pos = len(paper_content)
        for title in sections_to_remove:
            pattern_str = f"^#{{1,3}}\\s*(\\d+\\.?\\s+)?{re.escape(title)}\\b"
            match = re.search(pattern_str, paper_content, re.IGNORECASE | re.MULTILINE)
            if match and match.start() < cut_off_pos:
                cut_off_pos = match.start()
        
        cleaned_paper = paper_content[:cut_off_pos].strip()

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_paper)
        
        if pdf_to_delete_path.exists():
            os.remove(pdf_to_delete_path)
        
        return md_path, None
    except Exception as e:
        return None, f"清理Markdown文件时出错: {e}"

# --- 步骤 4: 抓取元数据并生成总结 (单文件版本) ---
def create_paper_json(arxiv_id, md_path, db_save_path):
    """抓取元数据，整合MD内容和总结，生成最终的JSON文件。"""
    try:
        arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        api_url = f'https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}'
        api_params = {'fields': 'citationCount,influentialCitationCount,authors.name,authors.affiliations,venue,publicationVenue,publicationDate,fieldsOfStudy,title'}
        
        api_response = requests.get(api_url, headers=headers, params=api_params, timeout=15)
        api_response.raise_for_status()
        api_data = api_response.json()
        
        authors_list = [author.get('name', '未知作者') for author in api_data.get('authors', [])]
        institutions = sorted(list(set(aff for author in api_data.get('authors', []) for aff in author.get('affiliations', []) if aff and aff.strip())))

        paper_info = {
            'link': arxiv_url,
            'pdf_link': arxiv_url.replace('/abs/', '/pdf/'),
            'title': api_data.get('title', '未知标题'),
            'authors': authors_list,
            'institutions': institutions or ["未找到机构信息"],
            'publication_date': api_data.get('publicationDate'),
            'venue': api_data.get('venue') or (api_data.get('publicationVenue') or {}).get('name') or "未找到",
            'fields_of_study': api_data.get('fieldsOfStudy', []),
            'citation_count': api_data.get('citationCount', 0),
            'influential_citation_count': api_data.get('influentialCitationCount', 0),
        }

        with open(md_path, 'r', encoding='utf-8') as f:
            paper_info['paper_content'] = f.read()

        # 调用我们模拟的总结函数
        paper_info['summary'] = summary_func(arxiv_id) 

        # 根据你的新要求，数据库路径现在包含日期和分类
        # 这里我们假设它来自一个通用处理流程，先不加分类
        db_save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(db_save_path, 'w', encoding='utf-8') as f:
            json.dump(paper_info, f, ensure_ascii=False, indent=4)
            
        return db_save_path, paper_info, None

    except Exception as e:
        return None, None, f"创建JSON文件时出错: {e}"

# =================================================================
# 3. Streamlit UI 界面和逻辑
# =================================================================

# --- 辅助UI函数 ---
def display_paper_summary(paper_data, container=st):
    """
    在指定容器（如st.expander）中以美化格式展示单篇论文的摘要和元数据。
    - 包含一个醒目的标题。
    - 修复了字段错误，并美化了链接显示。
    - 【新功能】自动修复并渲染LaTeX公式。
    - 【新功能】美化摘要的展示结构。
    """
    if not paper_data or not paper_data.get('title'):
        return

    # --- 元数据展示部分 ---
    container.markdown(f"### 📄 {paper_data.get('title', '未知标题')}")
    
    arxiv_id = paper_data.get('link', 'N/A').split('/')[-1]

    col1, col2 = container.columns([2, 1]) 

    with col1:
        container.markdown(f"**✍️ 作者:** {', '.join(paper_data.get('authors', ['未找到作者信息']))}")
        institutions = paper_data.get('institutions', ["未找到机构信息"])
        container.markdown(f"**🏢 机构:** {', '.join(institutions)}")
        container.markdown(f"**🗓️ 发表日期:** {paper_data.get('publication_date', '未找到发表日期')}")
    
    with col2:
        container.markdown(f"**🆔 ArXiv ID:** `{arxiv_id}`")
        container.markdown(f"**📈 引用数:** {paper_data.get('citation_count', '未找到引用量')} | **高影响力引用:** {paper_data.get('influential_citation_count', '未找到高影响力引用量')}")
        
        arxiv_link = paper_data.get('link')
        pdf_link = paper_data.get('pdf_link')
        
        if arxiv_link:
            container.markdown(f"**🔗 ArXiv页面:** [点击跳转]({arxiv_link})")
        if pdf_link:
            container.markdown(f"**📥 PDF下载:** [直接下载]({pdf_link})")
            
    container.divider()
    container.subheader("论文总结")

    summary_content = paper_data.get('summary')
    if summary_content:
        try:
            # 步骤 1: 尝试解析JSON字符串
            summary_content_fixed_escape = summary_content.replace("\\", "\\\\") 
            summary_dict = json.loads(summary_content_fixed_escape)
            
            if isinstance(summary_dict, dict):
                # 步骤 2: 遍历摘要的每个部分，并进行美化渲染
                for key, value in summary_dict.items():

                    # 步骤 3: 智能修复并渲染公式
                    # 正则表达式，查找被圆括号包裹、且内部含有'\'的LaTeX公式
                    # 这可以避免错误地将普通括号内容 (like this) 转换为公式
                    latex_pattern = r'\(([^)]*\\.*?)\)'
                    # 将找到的 (...) 格式替换为 $...$ 格式
                    # r'$\1$' 中的 \1 代表正则表达式中第一个括号 (...) 捕获的内容
                    value_with_fixed_latex = re.sub(latex_pattern, r'$\1$', value)

                    # 步骤 4: 处理换行符，使其在Markdown中生效
                    # 将 '\n' 替换为 '  \n' (两个空格+换行符) 来创建硬换行
                    final_value = value_with_fixed_latex.replace("\\n", "  \n")
                    
                    # 使用带边框的容器来展示内容，增加美感
                    with container.container(border=True):
                        st.markdown(final_value, unsafe_allow_html=True)

            else:
                # 如果解析出来不是字典，则按原样显示
                container.write(summary_content)
        
        except (json.JSONDecodeError, TypeError):
            # 如果summary不是有效的JSON字符串，则直接将其作为普通文本显示
            # 同样在这里尝试修复公式
            latex_pattern = r'\(([^)]*\\.*?)\)'
            fixed_content = re.sub(latex_pattern, r'$\1$', summary_content)
            container.markdown(fixed_content)
            
    else:
        container.info("此论文暂无总结信息。")

def display_paper_card(paper_data):
    """
    用 st.expander 展示一篇论文的卡片，包含一个按钮用于跳转到详情页。
    """
    if not paper_data or not paper_data.get('title'):
        return

    expander_title = paper_data.get('title', '未知标题')
    
    with st.expander(f"**{expander_title}**", expanded=False):
        # 点击后，设置 session_state 以切换到详情页视图
        if st.button("深入阅读与对话", key=f"detail_{paper_data['link']}"):
            st.session_state.current_view = 'detail_view'
            st.session_state.active_paper_data = paper_data
            # 清空上一篇论文的聊天记录
            if 'chat_messages' in st.session_state:
                st.session_state.chat_messages.clear()
            st.rerun()
        
        display_paper_summary(paper_data, container=st)


# --- 页面渲染函数 ---

def render_main_page():
    """渲染主欢迎页面"""
    st.title("🚀 欢迎使用 PaperSAGE")
    st.markdown("请从左侧侧边栏选择一项功能开始。")
    st.image("https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?q=80&w=2070", caption="探索科学的前沿")


def render_smart_summary_page():
    """渲染“论文总结”页面"""
    st.header("✍️ 论文总结生成")
    st.markdown("输入一篇ArXiv论文的链接，我们将为您下载、解析并生成一份论文总结。")
    
    arxiv_url = st.text_input("请输入ArXiv论文链接", key="arxiv_url_input")

    if st.button("开始处理", type="primary"):
        if not arxiv_url:
            st.error("请输入有效的ArXiv链接。")
        elif '请' in API_TOKEN or len(API_TOKEN) < 20:
            st.error("错误：请先在脚本中配置您的有效 API_TOKEN。")
        else:
            today_str = datetime.now().strftime("%Y%m%d")
            origin_dir = BASE_PATH / "origin_papers" / today_str
            transferred_dir = BASE_PATH / "transferred_papers" / today_str
            db_dir = DB_ROOT_PATH / today_str  # 注意路径变化
            
            origin_dir.mkdir(parents=True, exist_ok=True)
            transferred_dir.mkdir(parents=True, exist_ok=True)
            
            arxiv_id = arxiv_url.split('/')[-1]
            final_json_path = db_dir / f"{arxiv_id}.json"

            with st.status("正在处理论文...", expanded=True) as status:
                st.write("1/5: 正在下载 PDF...")
                pdf_path, error = download_arxiv_pdf(arxiv_url, origin_dir)
                if error:
                    status.update(label=f"错误: {error}", state="error"); return

                st.write("2/5: 正在转换 PDF 为 Markdown...")
                md_path, error = convert_single_pdf_to_md(pdf_path, transferred_dir)
                if error:
                    status.update(label=f"错误: {error}", state="error"); return

                st.write("3/5: 正在清理 Markdown 内容...")
                cleaned_md_path, error = process_single_md_file(md_path, pdf_path)
                if error:
                    status.update(label=f"错误: {error}", state="error"); return

                st.write("4/5: 正在抓取元数据并生成总结...")
                # 注意：这里我们假设处理的论文没有预先分类，所以保存在日期目录下
                json_path, final_data, error = create_paper_json(arxiv_id, cleaned_md_path, final_json_path)
                if error:
                    status.update(label=f"错误: {error}", state="error"); return

                st.write(f"5/5: 最终 JSON 文件已保存到: {json_path}")
                status.update(label="🎉 处理完成！", state="complete")

            st.subheader("处理结果预览")
            if final_data:
                display_paper_card(final_data)


def render_daily_recommendation_page():
    """渲染“每日推荐”页面"""
    st.header("📅 每日推荐")
    st.markdown("选择您感兴趣的领域，我们将为您推荐今天该领域的最新论文。")
    
    # 假设你的数据库结构是 /database/YYYYMMDD/cs.AI/*.json
    today_str = datetime.now().strftime("%Y%m%d")
    # 为了演示，如果今天没有，就用前一天的
    date_to_check = DB_ROOT_PATH / today_str
    if not date_to_check.exists():
        yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        date_to_check = DB_ROOT_PATH / yesterday_str
        if date_to_check.exists():
            st.info(f"未找到今日({today_str})数据，显示昨日({yesterday_str})推荐。")
        else:
            st.warning(f"数据库中未找到近期论文 ({today_str} 或 {yesterday_str})。请先处理一些论文。")
            return

    # 让用户选择分类
    top_category = st.selectbox("选择一个大学科", list(ARXIV_CATEGORIES.keys()))
    sub_category = st.selectbox("选择一个子领域", ARXIV_CATEGORIES[top_category])

    category_path = date_to_check / sub_category
    if not category_path.exists() or not any(category_path.iterdir()):
        st.info(f"在 {date_to_check.name}/{sub_category} 目录下没有找到论文。")
        return

    # 从分类文件夹加载论文
    if 'daily_papers' not in st.session_state or st.session_state.get('current_category') != sub_category:
        st.session_state.all_daily_papers = list(category_path.glob("*.json"))
        st.session_state.current_category = sub_category
        st.session_state.daily_papers = random.sample(
            st.session_state.all_daily_papers, 
            min(10, len(st.session_state.all_daily_papers))
        )
    
    if st.button("🔄 换一批"):
        st.session_state.daily_papers = random.sample(
            st.session_state.all_daily_papers,
            min(10, len(st.session_state.all_daily_papers))
        )
        st.rerun()

    st.markdown(f"--- \n ### 为您从 **{sub_category}** 领域推荐了 **{len(st.session_state.daily_papers)}** 篇论文")

    selected_papers = {}
    for paper_path in st.session_state.daily_papers:
        try:
            with open(paper_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                is_selected = st.checkbox(f"**{data.get('title', '无标题')}**", key=f"select_{paper_path.name}")
                if is_selected:
                    selected_papers[paper_path.name] = data
        except Exception as e:
            st.error(f"加载文件 {paper_path.name} 出错: {e}")
    
    if st.button("查看选中论文的总结", type="primary"):
        if not selected_papers:
            st.warning("请至少选择一篇论文。")
        else:
            st.subheader("所选论文详情")
            for name, data in selected_papers.items():
                display_paper_card(data)


def render_guess_you_like_page():
    """渲染“猜你喜欢”页面 (目前为随机推荐)"""
    st.header("❤️ 猜你喜欢")
    st.markdown("根据我们的数据库，为您随机推荐一些可能感兴趣的论文。")
    
    # 搜索框 (功能待实现)
    query = st.text_input("输入关键词进行搜索（功能开发中...）")

    if not DB_ROOT_PATH.exists() or not any(DB_ROOT_PATH.glob("**/*.json")):
        st.warning(f"数据库 '{DB_ROOT_PATH}' 为空。请先处理一些论文。")
        return

    if 'all_db_papers' not in st.session_state:
        st.session_state.all_db_papers = list(DB_ROOT_PATH.glob("**/*.json"))
    
    if 'recommended_papers' not in st.session_state:
        st.session_state.recommended_papers = random.sample(
            st.session_state.all_db_papers, 
            min(5, len(st.session_state.all_db_papers))
        )

    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("🔍 搜索/换一批", type="primary"):
            if query:
                st.info(f"搜索功能正在开发中，将为您随机推荐。您搜索了：'{query}'")
            st.session_state.recommended_papers = random.sample(
                st.session_state.all_db_papers,
                min(5, len(st.session_state.all_db_papers))
            )
            st.rerun()

    if st.session_state.recommended_papers:
        for paper_path in st.session_state.recommended_papers:
            try:
                with open(paper_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    display_paper_card(data)
            except Exception as e:
                st.error(f"加载或显示文件 {paper_path.name} 出错: {e}")


def render_detail_view():
    """渲染论文详情对话页面"""
    if 'active_paper_data' not in st.session_state:
        st.error("无法加载论文数据，请返回主页重试。")
        if st.button("返回"):
            st.session_state.current_view = "main_page"
            st.rerun()
        return

    paper_data = st.session_state.active_paper_data
    st.title(paper_data.get('title', '论文详情'))
    
    if st.button("⬅️ 返回列表"):
        # 根据用户从哪里进入详情页，决定返回到哪个页面
        # 简单起见，我们统一返回到“猜你喜欢”
        st.session_state.current_view = 'guess_you_like' 
        del st.session_state['active_paper_data']
        if 'chat_messages' in st.session_state:
            del st.session_state['chat_messages']
        st.rerun()

    st.markdown(f"阅读原文与AI对话：**[{paper_data.get('title')}]({paper_data.get('link')})**")
    st.divider()

    col1, col2 = st.columns([1, 1]) # 左右等宽

    # 左侧：Markdown原文和可点击的标题
    with col1:
        st.header("📄 论文原文")
        content = paper_data.get('paper_content', '无原文内容。')
        
        # 使用正则表达式查找所有Markdown标题
        header_pattern = re.compile(r"(^#+\s+.*)", re.MULTILINE)
        parts = header_pattern.split(content)
        
        # 显示非标题部分
        st.markdown(parts[0], unsafe_allow_html=True)
        
        # 显示标题和其后的内容
        for i in range(1, len(parts), 2):
            header = parts[i]
            section_content = parts[i+1]
            
            # 创建一个可点击的标题按钮
            if st.button(header, use_container_width=True):
                # 点击后，调用翻译函数并将结果添加到聊天记录
                translated_text = translate_text(header + '\n' + section_content.split('\n\n')[0]) # 仅翻译标题和第一段
                if 'chat_messages' not in st.session_state:
                    st.session_state.chat_messages = []
                st.session_state.chat_messages.append({"role": "user", "content": f"请翻译这个部分: {header}"})
                st.session_state.chat_messages.append({"role": "assistant", "content": translated_text})

            st.markdown(section_content, unsafe_allow_html=True)

    # 右侧：对话框
    with col2:
        st.header("💬 与论文对话")
        
        # 初始化聊天记录
        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages = []
        
        # 显示历史消息
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 接收用户输入
        if prompt := st.chat_input("就这篇论文进行提问..."):
            # 将用户消息添加到历史记录并显示
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 生成并显示AI的回答
            with st.chat_message("assistant"):
                response = rag_query(prompt, content) # 调用RAG函数
                st.markdown(response)
            
            # 将AI回答添加到历史记录
            st.session_state.chat_messages.append({"role": "assistant", "content": response})

# =================================================================
# 4. 主应用入口
# =================================================================

# --- 初始化 Session State ---
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'main_page' # 默认显示主页

# --- 侧边栏导航 ---
with st.sidebar:
    st.title("📚 PaperSAGE 导航")
    
    # 使用 on_change 回调来清除特定状态，防止页面切换时数据混乱
    def change_page():
        # 当我们离开详情页时，清除活动论文数据
        if st.session_state.current_view == 'detail_view':
             if 'active_paper_data' in st.session_state:
                del st.session_state['active_paper_data']
             if 'chat_messages' in st.session_state:
                del st.session_state['chat_messages']
        # 将 radio 的选择同步到 current_view
        st.session_state.current_view = st.session_state.page_selection

    page = st.radio(
        "选择功能",
        options=['main_page', 'smart_summary', 'daily_recommendation', 'guess_you_like'],
        format_func=lambda x: {
            'main_page': '🏠 主页',
            'smart_summary': '✍️ 论文总结',
            'daily_recommendation': '📅 每日推荐',
            'guess_you_like': '❤️ 猜你喜欢'
        }.get(x, x),
        key='page_selection',
        on_change=change_page
    )

    st.sidebar.divider()
    st.sidebar.info("PaperSAGE - 您的智能论文阅读助手。")


# --- 页面调度器 ---
if st.session_state.current_view == 'main_page':
    render_main_page()
elif st.session_state.current_view == 'smart_summary':
    render_smart_summary_page()
elif st.session_state.current_view == 'daily_recommendation':
    render_daily_recommendation_page()
elif st.session_state.current_view == 'guess_you_like':
    render_guess_you_like_page()
elif st.session_state.current_view == 'detail_view':
    render_detail_view()
else:
    render_main_page() # 默认或出错时返回主页
