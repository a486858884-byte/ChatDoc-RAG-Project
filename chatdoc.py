# ===================================================================
# 文件名: chatdoc.py
# 版本: Agentic RAG Final (集成 LangGraph + 熔断机制 + MySQL)
# ===================================================================

import os
import sys
import getpass
from typing import TypedDict, List
from dotenv import load_dotenv

# --- 数据库组件 ---
import mysql.connector
from mysql.connector import Error

# --- LangChain / LangGraph 核心组件 ---
from langchain.chat_models import init_chat_model
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# ===================================================================
# 1. 环境与基础设施初始化 (Infrastructure)
# ===================================================================
print("=== 系统启动中 ===")

# 1.1 加载环境变量
script_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv()
cache_dir = os.path.join(script_dir, "models")
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_dir

if not os.environ.get("DEEPSEEK_API_KEY"):
    print("⚠️ 警告：未检测到 DEEPSEEK_API_KEY，请检查 .env 文件")


# 1.2 初始化数据库连接
def create_db_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("✅ MySQL 数据库连接成功")
    except Error as e:
        print(f"⚠️ 数据库连接失败 (仅影响记录存储): {e}")
    return connection


def execute_read_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        return cursor.fetchall()
    except Error as e:
        print(f"查询错误: {e}")


def execute_write_query(connection, query, data=None):
    cursor = connection.cursor()
    try:
        cursor.execute(query, data)
        connection.commit()
    except Error as e:
        print(f"写入错误: {e}")


# 连接数据库 & 初始化用户
db_conn = create_db_connection("localhost", "root", "1234", "chatdoc_db")
current_user_id = None
if db_conn:
    current_username = "alice"
    # 简单的用户检查/创建逻辑
    user_in_db = execute_read_query(db_conn, f"SELECT id FROM users WHERE username = '{current_username}'")
    if user_in_db:
        current_user_id = user_in_db[0][0]
    else:
        execute_write_query(db_conn, f"INSERT INTO users (username) VALUES ('{current_username}')")
        user_in_db = execute_read_query(db_conn, f"SELECT id FROM users WHERE username = '{current_username}'")
        if user_in_db: current_user_id = user_in_db[0][0]

# 1.3 初始化模型 (LLM & Embeddings)
print("正在加载 AI 模型...")
llm = init_chat_model("deepseek-chat", model_provider="deepseek", temperature=0.1)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-zh-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 1.4 初始化向量知识库 (FAISS)
print("正在加载知识库...")
INDEX_PATH = os.path.join(script_dir, "faiss_index_store")
retriever = None

try:
    if os.path.exists(INDEX_PATH):
        print("✅ 加载本地索引...")
        vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever()
    else:
        print("⚡ 构建新索引 (耗时操作)...")
        loader = TextLoader(os.path.join(script_dir, "sample.txt"), encoding="utf-8")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)
        vector_store = FAISS.from_documents(split_docs, embeddings)
        vector_store.save_local(INDEX_PATH)
        retriever = vector_store.as_retriever()
except Exception as e:
    print(f"❌ 知识库加载严重错误: {e}")


# ===================================================================
# 2. Agent 大脑构建区 (The Brain)
# ===================================================================

# 2.1 定义共享内存 (State)
class GraphState(TypedDict):
    question: str  # 用户原始问题
    generation: str  # 生成的答案
    documents: List[Document]  # 文档列表
    grade: str  # 评分 yes/no
    search_query: str  # 搜索词
    loop_step: int  # 熔断计数器


# 2.2 定义工具链 (Chains)

# A. 质检链 (Grader)
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="文档相关性评分，'yes' 或 'no'")


structured_llm_grader = llm.with_structured_output(GradeDocuments)
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个严厉的阅卷老师。评估文档片段是否与问题相关。相关评'yes'，不相关评'no'。"),
    ("human", "问题: {question} \n\n 文档: {document}"),
])
retrieval_grader = grade_prompt | structured_llm_grader

# B. 生成链 (Writer)
# --- B. 生成链 (Writer) - 修正版 ---
generate_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个完全依赖上下文的严谨 AI 助手。
    1. 你只能根据提供的【上下文】回答问题。
    2. **严禁**使用你的预训练知识或外部知识。
    3. 如果【上下文】中找不到答案，或者上下文是空的，你必须直接回答：“抱歉，当前知识库中没有关于此问题的记录。”
    4. 不要编造，不要尝试去定义你自己，除非文档里明确写了。"""),
    ("human", "上下文: \n{context} \n\n 问题: {question}"),
])
rag_chain = generate_prompt | llm | StrOutputParser()

# C. 改写链 (Rewriter)
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是搜索优化专家。将用户问题改写为更精准的搜索关键词。只输出改写后的内容。"),
    ("human", "原始输入: {question}"),
])
question_rewriter = rewrite_prompt | llm | StrOutputParser()


# 2.3 定义节点函数 (Nodes)

def retrieve(state: GraphState):
    """节点：检索"""
    print("---NODE: RETRIEVE---")
    query = state["search_query"]
    documents = retriever.invoke(query)
    print(f"  => 搜到 {len(documents)} 条文档")
    return {"documents": documents}


def grade_documents(state: GraphState):
    """节点：质检"""
    print("---NODE: GRADE---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score == "yes":
            filtered_docs.append(d)

    global_grade = "yes" if len(filtered_docs) > 0 else "no"
    print(f"  => 质检结果: {global_grade} (保留 {len(filtered_docs)} 条)")
    return {"documents": filtered_docs, "grade": global_grade}


def generate(state: GraphState):
    """节点：生成"""
    print("---NODE: GENERATE---")
    question = state["question"]
    documents = state["documents"]
    context = "\n\n".join([d.page_content for d in documents])
    generation = rag_chain.invoke({"context": context, "question": question})
    return {"generation": generation}


def rewrite_query(state: GraphState):
    """节点：改写 (带熔断监控)"""
    print("---NODE: REWRITE---")
    try:
        current_step = state.get("loop_step", 0)
        print(f"  [DEBUG] 重试计数: {current_step}")

        better_query = question_rewriter.invoke({"question": state["question"]})
        print(f"  => 优化搜索词: {better_query}")

        return {"search_query": better_query, "loop_step": current_step + 1}
    except Exception as e:
        print(f"❌ Rewrite 节点出错: {e}")
        return {"search_query": state["question"], "loop_step": state.get("loop_step", 0) + 1}


def decide_to_generate(state):
    """路由逻辑 (带熔断)"""
    print("---DECISION---")
    grade = state["grade"]
    current_step = state.get("loop_step", 0)

    if current_step >= 3:
        print("  => ⚠️ 触发熔断 (Max Retries)，强制生成。")
        return "generate"

    if grade == "yes":
        return "generate"
    else:
        print("  => 质量不达标，尝试改写...")
        return "rewrite_query"


# 2.4 组装图 (Workflow)
app = None
if retriever:
    print("正在组装 Agent 神经网络...")
    workflow = StateGraph(GraphState)

    # 添加节点
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("rewrite_query", rewrite_query)

    # 连线
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)

    # 条件边
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query"
        }
    )

    app = workflow.compile()
    print("✅ Agent 编译完成！")
else:
    print("❌ Agent 组装失败：Retriever 未就绪")


# ===================================================================
# 3. 对外接口 (API Interface)
# ===================================================================

def get_rag_response(user_input):
    """
    统一入口：接收用户问题 -> 运行 Agent -> 存库 -> 返回答案
    """
    if not app:
        return "系统错误：Agent 未就绪。"

    try:
        # 1. 构造输入 (初始化熔断计数器)
        inputs = {
            "question": user_input,
            "search_query": user_input,
            "loop_step": 0
        }

        print(f"\n🚀 [Agent 启动] 问题: {user_input}")

        # 2. 运行图
        final_state = app.invoke(inputs)

        # 3. 提取结果
        ai_answer = final_state.get("generation", "抱歉，未能生成回答。")
        retry_count = final_state.get("loop_step", 0)

        # 4. 存入数据库
        if db_conn and current_user_id:
            insert_query = "INSERT INTO conversations (user_id, user_message, ai_response) VALUES (%s, %s, %s);"
            execute_write_query(db_conn, insert_query, (current_user_id, user_input, ai_answer))

        # (可选) 在控制台展示一下有没有触发重试
        if retry_count > 0:
            print(f"✨ 本次回答触发了 {retry_count} 次自我修正。")

        return ai_answer

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"运行出错: {str(e)}"


# ===================================================================
# 4. 命令行测试入口
# ===================================================================
if __name__ == '__main__':
    print("\n✅ 系统已就绪。进入命令行测试模式...")
    print("提示：尝试输入模糊问题触发重试，如 '那玩意儿要预热多久？'")

    while True:
        try:
            q = input("\n请输入问题 (quit退出): ")
            if q.lower() in ['quit', 'exit']:
                break

            # 调用封装好的接口
            answer = get_rag_response(q)
            print(f"\n🤖 AI 回答:\n{answer}")

        except KeyboardInterrupt:
            break

    if db_conn:
        db_conn.close()
        print("数据库连接已关闭。")