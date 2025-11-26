# 1. 准备工作：导入所有需要的“建筑材料”（库）
import os
import sys
import getpass
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error

# --- LangChain 核心组件 ---
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# ===================================================================
# 核心环境变量设置
# ===================================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(script_dir, "models")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir


# ===================================================================
# 数据库辅助函数 (保持不变)
# ===================================================================
def create_db_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("MySQL 数据库连接成功！")
    except Error as e:
        print(f"连接失败，错误信息: '{e}'")
    return connection


def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as e:
        print(f"查询失败，错误信息: '{e}'")


def execute_write_query(connection, query, data=None):
    cursor = connection.cursor()
    try:
        cursor.execute(query, data)
        connection.commit()
        # print("查询执行成功！") # 注释掉，避免刷屏
    except Error as e:
        print(f"查询失败，错误信息: '{e}'")


# ===================================================================
# !!! 全局初始化 (从 main 里搬出来的，顶格写) !!!
# 这样 app.py 导入时，这些代码会自动运行，变量就能被使用了
# ===================================================================

print("=== 系统初始化开始 ===")

# 1. 加载 API
load_dotenv()
if not os.environ.get("DEEPSEEK_API_KEY"):
    # 注意：在 Streamlit 环境下，input/getpass 可能无法交互，最好确保 .env 里有 key
    print("警告：未找到 API Key，请确保 .env 文件配置正确")

# 2. 初始化模型
print("正在初始化模型...")
llm = init_chat_model("deepseek-chat", model_provider="deepseek", temperature=0.1)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 3. 构建知识库
print("正在加载知识库...")

# 定义一个存索引的文件夹名字
# os.path.join 保证了不管你在 Windows 还是 Linux 都能跑
INDEX_PATH = os.path.join(script_dir, "faiss_index_store")

try:
    # --- 分支一：检查本地有没有索引文件 ---
    if os.path.exists(INDEX_PATH):
        print("✅ 发现本地索引，正在直接加载 (省流模式)...")

        # 【关键点】 allow_dangerous_deserialization=True
        # 新版 LangChain 为了安全（防止加载恶意 pickle 文件）加的锁。
        # 因为是你自己生成的文件，自己信自己，设为 True 没问题。
        vector_store = FAISS.load_local(
            INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = vector_store.as_retriever()

    # --- 分支二：本地没有，老老实实去算 ---
    else:
        print("⚡ 未发现本地索引，正在重新构建 (这将消耗 Token)...")

        # 这一段是你原来的逻辑
        loader = TextLoader(os.path.join(script_dir, "sample.txt"), encoding="utf-8")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)

        vector_store = FAISS.from_documents(split_docs, embeddings)

        # 【关键点】 算完立刻存盘！
        vector_store.save_local(INDEX_PATH)
        print(f"✅ 索引已保存到: {INDEX_PATH}")

        retriever = vector_store.as_retriever()

except Exception as e:
    print(f"❌ 知识库加载失败: {e}")
    # 如果出错（比如文件夹坏了），你可能得手动删了那个文件夹重跑
    retriever = None

# 4. 创建 RAG 链
if retriever:
    prompt = ChatPromptTemplate.from_template("""
    你是一个逻辑严密的 AI 助手。请根据下面的上下文回答问题。在回答之前，请先进行**一步步的逻辑推理 (Chain of Thought)**，引用上下文中的具体证据。
    <context>{context}</context>
    问题: {input}
    """)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print("AI 处理链创建完成！")
else:
    retrieval_chain = None

# 5. 初始化数据库连接
print("正在连接数据库...")
# !!! 请确保这里的密码是正确的 !!!
db_conn = create_db_connection("localhost", "root", "1234", "chatdoc_db")

# 6. 初始化用户 (硬编码 Alice)
current_user_id = None
if db_conn:
    current_username = "alice"
    user_in_db = execute_read_query(db_conn, f"SELECT id FROM users WHERE username = '{current_username}'")
    if user_in_db:
        current_user_id = user_in_db[0][0]
        print(f"用户 {current_username} 已加载 (ID: {current_user_id})")
    else:
        print(f"创建新用户 {current_username}...")
        execute_write_query(db_conn, f"INSERT INTO users (username) VALUES ('{current_username}')")
        user_in_db = execute_read_query(db_conn, f"SELECT id FROM users WHERE username = '{current_username}'")
        if user_in_db:
            current_user_id = user_in_db[0][0]

print("=== 系统初始化完成 ===")


# ===================================================================
# !!! 核心接口函数 (供 Streamlit 调用) !!!
# ===================================================================
def get_rag_response(user_input):
    """
    接收用户问题，调用 RAG 链，存入数据库，返回答案
    """
    if not retrieval_chain:
        return "错误：知识库未正确初始化，请检查后台日志。"

    try:
        # 1. AI 思考
        response = retrieval_chain.invoke({"input": user_input})
        print(f"\n[DEBUG] 用户问题: {user_input}")

        # 把检索到的文档片段打印出来看看！
        # source_documents 或者 context，取决于你的链怎么建的，通常是 context
        context_docs = response.get("context", [])
        print(f"[DEBUG] 检索到的参考文档 ({len(context_docs)} 条):")
        for i, doc in enumerate(context_docs):
            # page_content 就是切分后的那 500 字原文
            print(f"  --- 文档片段 {i + 1} ---")
            print(f"  {doc.page_content[:100]}...")  # 只打前100字看看
        ai_answer = response["answer"]

        # 2. 存入数据库 (如果有连接)
        if db_conn and current_user_id:
            insert_query = """
            INSERT INTO conversations (user_id, user_message, ai_response) 
            VALUES (%s, %s, %s);
            """
            data = (current_user_id, user_input, ai_answer)
            execute_write_query(db_conn, insert_query, data)

        return ai_answer

    except Exception as e:
        return f"发生错误: {str(e)}"


# ===================================================================
# 命令行测试入口 (仅当直接运行 chatdoc.py 时执行)
# ===================================================================
if __name__ == '__main__':
    print("\n进入命令行测试模式...")
    while True:
        q = input("请输入问题 (输入 quit 退出): ")
        if q.lower() in ['quit', 'exit']:
            break
        print("回答:", get_rag_response(q))