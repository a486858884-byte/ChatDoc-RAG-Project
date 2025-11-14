# 1. 准备工作：导入所有需要的“建筑材料”（库）
# ===================================================================
# ===================================================================
# 核心环境变量设置 (必须放在所有导入之前)
# ===================================================================
import os
import sys # 确保导入 sys

# 1. 获取当前脚本文件所在的目录的绝对路径
#    __file__ 是一个 Python 内置变量，代表当前脚本的文件名
#    os.path.dirname() 获取该文件所在的目录
#    os.path.abspath() 确保我们得到的是绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

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


# --- 基础工具 ---
import os
import getpass
from dotenv import load_dotenv

# --- LangChain 核心组件 ---
# LLM 初始化
from langchain.chat_models import init_chat_model
# Embedding 模型
from langchain_huggingface import HuggingFaceEmbeddings
# 文档加载器
from langchain_community.document_loaders import TextLoader
# 文本分割器
from langchain_text_splitters import RecursiveCharacterTextSplitter
# 向量数据库
from langchain_community.vectorstores import FAISS
# 提示模板
from langchain_core.prompts import ChatPromptTemplate
# 链的组装工具
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# --- 数据库连接组件 ---
import mysql.connector
from mysql.connector import Error

# 功能一：建立与数据库的连接
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
# 功能二：执行“读取”类型的查询 (SELECT)
def execute_read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as e:
        print(f"查询失败，错误信息: '{e}'")
# 功能三：执行“写入”类型的查询 (INSERT, UPDATE, DELETE)
# --- 修改这个函数 ---
def execute_write_query(connection, query, data=None):
    cursor = connection.cursor()
    try:
        # 将数据和查询一起执行
        cursor.execute(query, data)
        connection.commit()
        print("查询执行成功！")
    except Error as e:
        print(f"查询失败，错误信息: '{e}'")


# ===================================================================
# 主函数：程序的总入口
# ===================================================================
if __name__ == '__main__':
    # --- 1. 初始化所有核心组件 (AI 部分) ---
    # (这部分代码和原来一样，只是放到了主函数里)
    print("正在加载 API 密钥...")
    load_dotenv()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("未在 .env 文件中找到密钥，请输入您的 DeepSeek API Key: ")
    print("API 密钥加载成功。")

    print("正在初始化模型...")
    llm = init_chat_model("deepseek-chat", model_provider="deepseek", temperature=0.1)
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-zh-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("模型初始化完成。")

    print("正在加载文档并构建知识库...")
    loader = TextLoader("sample.txt", encoding="utf-8")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    retriever = vector_store.as_retriever()
    print("知识库构建完成。")

    prompt = ChatPromptTemplate.from_template("""
    请只根据下面提供的上下文来回答用户的问题...
    <context>{context}</context>
    问题: {input}
    """)
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print("处理链创建完成，系统准备就绪！")

    # --- 2. [新增] 初始化数据库连接 ---
    print("\n正在连接到 MySQL 数据库...")
    # !!! 再次确认，将 'Your_Password_Here' 替换为您的真实密码 !!!
    db_conn = create_db_connection("localhost", "root", "1234", "chatdoc_db")

    # 如果连接失败，则直接退出程序
    if not db_conn:
        print("无法连接到数据库，程序退出。")
        exit()

    # --- 3. [新增] 获取或创建用户 ---
    # 为了简化，我们先硬编码一个用户名
    current_username = "alice"
    # 尝试在数据库中查找这个用户
    user_in_db = execute_read_query(db_conn, f"SELECT id FROM users WHERE username = '{current_username}'")

    if user_in_db:
        # 如果用户已存在，获取他的 id
        current_user_id = user_in_db[0][0]
        print(f"欢迎回来, {current_username} (用户 ID: {current_user_id})!")
    else:
        # 如果用户不存在，就创建一个新用户
        print(f"新用户 '{current_username}', 正在为您创建账户...")
        execute_write_query(db_conn, f"INSERT INTO users (username) VALUES ('{current_username}')")
        # 再次查询以获取新创建的 user_id
        user_in_db = execute_read_query(db_conn, f"SELECT id FROM users WHERE username = '{current_username}'")
        current_user_id = user_in_db[0][0]
        print(f"账户创建成功, {current_username} (用户 ID: {current_user_id})!")

    # --- 4. 改造交互式提问循环 ---
    print("\n" + "=" * 30)
    print("      欢迎来到 ChatDoc (数据库版)！")
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
        response = retrieval_chain.invoke({"input": user_question})
        ai_answer = response["answer"]

        print("\n回答:")
        print(ai_answer)

        # --- 5. [新增] 将对话记录存入数据库 ---
        # --- 在 while True 循环的末尾，修改这部分 ---
        print("正在保存对话记录...")

        # 创建带有占位符的 SQL 模板
        insert_convo_query = """
        INSERT INTO conversations (user_id, user_message, ai_response) 
        VALUES (%s, %s, %s);
        """

        # 准备要插入的数据，作为一个元组
        convo_data = (current_user_id, user_question, ai_answer)

        # 调用我们修改后的函数，将模板和数据分开传递
        execute_write_query(db_conn, insert_convo_query, convo_data)

    # --- 6. [新增] 在循环结束后，关闭数据库连接 ---
    if db_conn and db_conn.is_connected():
        db_conn.close()
        print("\nMySQL 连接已关闭。")