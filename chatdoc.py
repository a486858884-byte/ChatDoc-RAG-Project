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
print("正在加载文档并构建知识库...")
# 确保 sample.txt 路径正确，如果报错找不到文件，可能需要用绝对路径
try:
    loader = TextLoader(os.path.join(script_dir, "sample.txt"), encoding="utf-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    retriever = vector_store.as_retriever()
except Exception as e:
    print(f"知识库构建失败 (可能是文件路径问题): {e}")
    # 为了防止程序崩溃，给个空 retriever (仅用于调试)
    retriever = None

# 4. 创建 RAG 链
if retriever:
    prompt = ChatPromptTemplate.from_template("""
    请只根据下面提供的上下文来回答用户的问题...
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