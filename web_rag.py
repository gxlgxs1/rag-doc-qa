import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import streamlit as st
import requests
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
import tempfile

# ---------- 配置 ----------
API_KEY ="36beb600-4058-4e8e-b6de-8f2cf2330c2f"
MODEL_ID ="doubao-seed-2-0-code-preview-260215"   # 替换成你的接入点 ID
URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
HEADERS = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

# 初始化 Chroma 客户端（持久化）
client = chromadb.PersistentClient(path="./web_rag_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# 尝试获取或创建 collection
collection_name = "doc_collection"
try:
    collection = client.get_collection(name=collection_name, embedding_function=ef)
except:
    collection = client.create_collection(name=collection_name, embedding_function=ef)

# ---------- 辅助函数 ----------
def load_text_from_file(uploaded_file):
    """从上传的文件中提取文本（支持 txt 和 pdf）"""
    if uploaded_file.name.endswith('.txt'):
        return uploaded_file.read().decode('utf-8')
    elif uploaded_file.name.endswith('.pdf'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        reader = PdfReader(tmp_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        os.unlink(tmp_path)
        return text
    else:
        st.error("不支持的文件类型，请上传 .txt 或 .pdf")
        return None

def chunk_text(text, chunk_size=500, overlap=50):
    """将文本切分成多个片段"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # 尽量在句号处断开
            for i in range(min(end+20, len(text))-1, end-1, -1):
                if text[i] in '。！？\n':
                    end = i+1
                    break
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def add_to_vector_store(chunks, file_name):
    """将文档片段存入向量库，先清空该文档的旧数据（简单处理：删除整个collection重建）"""
    # 为了演示简单，每次新上传清空旧数据（实际可按文档名管理）
    global collection
    client.delete_collection(collection_name)
    collection = client.create_collection(name=collection_name, embedding_function=ef)
    ids = [f"{file_name}_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, ids=ids)
    st.success(f"已处理 {len(chunks)} 个片段，可开始提问")

def retrieve(query, top_k=3):
    results = collection.query(query_texts=[query], n_results=top_k)
    return results['documents'][0]

def ask_llm(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""请仅根据以下文档片段回答问题。如果文档中没有相关信息，请说“文档中未提及”。

文档片段：
{context}

问题：{query}

回答："""
    data = {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}]}
    response = requests.post(URL, headers=HEADERS, json=data)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"API错误: {response.status_code}"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="文档问答机器人", layout="wide")
st.title("📄 文档智能问答 (RAG)")
st.markdown("上传 PDF 或 TXT 文档，然后提问，AI 将基于文档内容回答。")

# 侧边栏：文件上传
with st.sidebar:
    st.header("📁 上传文档")
    uploaded_file = st.file_uploader("选择文件", type=["txt", "pdf"])
    if uploaded_file is not None:
        with st.spinner("正在读取并处理文档..."):
            text = load_text_from_file(uploaded_file)
            if text:
                chunks = chunk_text(text)
                st.info(f"文档已切分为 {len(chunks)} 个片段")
                add_to_vector_store(chunks, uploaded_file.name)

# 主区域：问答
st.header("💬 提问")
question = st.text_input("请输入你的问题：")
if question:
    if collection.count() == 0:
        st.warning("请先上传文档")
    else:
        with st.spinner("检索并生成答案..."):
            relevant = retrieve(question)
            st.caption(f"检索到 {len(relevant)} 个相关片段")
            answer = ask_llm(question, relevant)
            st.markdown(f"**答案：** {answer}")