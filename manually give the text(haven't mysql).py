import os
import getpass
from dotenv import load_dotenv
import sys
# --- [关键修改] ---
# 1. 获取当前脚本文件所在的目录的绝对路径
#    __file__ 是一个 Python 内置变量，代表当前脚本的文件名
#    os.path.dirname() 获取该文件所在的目录
script_dir = os.path.dirname(__file__)

# 2. 基于脚本目录，构建一个绝对的、唯一的模型缓存路径
#    os.path.join() 会智能地将路径拼接起来
cache_dir = os.path.join(script_dir, "models")

# 3. 设置环境变量，使用这个绝对路径
#    这样无论从哪里运行脚本，缓存路径都是固定的
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir
print(f"Hugging Face 镜像已设置为: {os.environ['HF_ENDPOINT']}")
print(f"模型缓存的绝对路径已设置为: {os.environ['SENTENCE_TRANSFORMERS_HOME']}")
# ----------------------------------------------------

# --- ChatDoc 项目施工蓝图 ---
# 1. 准备工作：导入所有需要的“建筑材料”（库）
from encodings import normalize_encoding
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain, split_list_of_docs
from langchain_classic.chains import create_retrieval_chain
import os
from dotenv import load_dotenv
import getpass

from openai import embeddings
from sentence_transformers.util import normalize_embeddings
from sympy.physics.units import temperature
from torch.nn.init import normal


# 2. 环境准备：加载并设置 API 密钥
load_dotenv()
if not os.environ.get("DEEPSEEK_API_KEY"):
    os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("文档中找不到，在这里输入：")
print("API Get OK")
# 3. 初始化核心组件：
#    - 3.1 初始化“图书管理员”（LLM - DeepSeek）
print("初始化DeepSeek Model")
llm = init_chat_model(
    "deepseek-chat",
    model_provider = "deepseek",
    temperature = 0.1
)
#    - 3.2 初始化“索引员”（Embedding Model - BGE）
model_name = "BAAI/bge-large-zh-v1.5"
print(f"加载模型: {model_name}")
embeddings = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = {'device': 'cpu'},
    encode_kwargs = {'normalize_embeddings':True}
)
# --- 知识库构建阶段 ---

# 4. 加载“书籍”：读取 sample.txt 文档
print("加载文档...")
loader = TextLoader("sample.txt",encoding="utf-8")
docs = loader.load()
print("文档加载ok")
# 5. “裁切”书籍：把长文档分割成小块
print("分割文本中...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,chunk_overlap=50
)
split_docs = text_splitter.split_documents(docs)
print(f"文档被分割成 {len(split_docs)} 个小块。")
# 6. “制作索引卡”并“存入文件柜”：创建向量数据库
print("正在创建向量数据库...")
vector_store = FAISS.from_documents(split_docs,embeddings)
print("向量数据库创建成功。")
# --- 智能问答阶段 ---

# 7. 准备“检索工具”：从向量数据库创建一个检索器
retriever = vector_store.as_retriever()
# 8. 设计“工作指令”：创建提示模板 (Prompt)
prompt = ChatPromptTemplate.from_template('''
根据提供的上下文进行回答，不要依赖外部的任何资料，只根据提供的资料进行回答，如果提供的资料无法给出答案
直接说“我无法回答这个答案”即可。

--------
{context}
--------

问题：{input}

''')
# 9. 组装“问答流水线”：将所有零件链接成一个 Retrieval Chain
#对于 document_chain，是 “谁来做 (llm)” 和 “怎么做 (prompt)”。
document_chain =create_stuff_documents_chain(llm,prompt)
#对于 retrieval_chain，是 “第一步做什么 (retriever)” 和 “第二步做什么 (document_chain)”。
retriever_chain = create_retrieval_chain(retriever,document_chain)
print("处理链创建完成，系统准备就绪！")
# 10. 启动应用：开始进行交互式提问
print("\n" + "=" * 30)
print("      欢迎来到 ChatDoc (DeepSeek 版)！")
print("      输入 '退出' 来结束对话。")
print("=" * 30)
while True:
    user_question = input("\n请输入您的问题: ")
    if user_question.strip().lower() in ["退出", "exit", "quit"]:
        print("感谢使用，再见！")
        break

    if not user_question.strip():
        continue
    print("正在思考...")
    response = retriever_chain.invoke({"input": user_question})
    print("\n回答:")
    print(response["answer"])