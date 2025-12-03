import os
import sys
from typing import TypedDict, List
from pydantic import BaseModel, Field

# --- LangChain / LangGraph 组件 ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# ===================================================================
# 1. 资源引用 (从你的 chatdoc.py 借东西)
# ===================================================================
try:
    # 这一步会执行 chatdoc.py 里的初始化代码，确保 LLM 和 FAISS 就绪
    from chatdoc import llm, retriever

    print("✅ 成功从 chatdoc 导入 LLM 和 Retriever")
except ImportError:
    print("❌ 错误：找不到 chatdoc.py 或导入失败。请确保两个文件在同一目录下。")
    sys.exit(1)


# ===================================================================
# 2. 定义状态字典 (The State)
# ===================================================================
class GraphState(TypedDict):
    """
    Agent 的共享内存
    """
    question: str  # 用户原始问题
    generation: str  # 生成的答案
    documents: List[Document]  # 检索到的文档列表
    grade: str  # 文档质量评分 ("yes" or "no")
    search_query: str  # 实际去搜的词 (可能会被改写)
    loop_step: int

# ===================================================================
# 3. 定义辅助工具 (Prompts & Grader)
# ===================================================================

# --- A. 质检员的打分卡 (Pydantic) ---
class GradeDocuments(BaseModel):
    """对检索到的文档进行相关性打分"""
    binary_score: str = Field(description="文档是否与问题相关？'yes' 或 'no'")


# 使用 DeepSeek 的结构化输出能力
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 质检 Prompt
system_prompt = """你是一个严厉的阅卷老师。
评估检索到的文档片段是否与用户问题相关。
如果文档包含解答线索，评为 'yes'，否则评为 'no'。"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "用户问题: {question} \n\n 文档片段: {document}"),
    ]
)
retrieval_grader = grade_prompt | structured_llm_grader

# --- B. 作家的写作 Prompt ---
generate_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是专业的 AI 助手。请根据提供的上下文回答问题。如果上下文不知所云，请诚实地说不知道。"),
        ("human", "上下文: \n{context} \n\n 问题: {question}"),
    ]
)
rag_chain = generate_prompt | llm | StrOutputParser()

# --- C. 策划的改写 Prompt ---
rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个搜索优化专家。请把用户的口语化问题改写为适合检索的关键词查询。只输出改写后的句子。"),
        ("human", "原始输入: {question}"),
    ]
)
question_rewriter = rewrite_prompt | llm | StrOutputParser()


# ===================================================================
# 4. 定义节点函数 (The Nodes)
# ===================================================================

def retrieve(state: GraphState):
    """节点：检索"""
    print("---NODE: RETRIEVE (检索中)---")
    query = state["search_query"]

    # 这里的 invoke 返回的是 List[Document]
    documents = retriever.invoke(query)
    print(f"  => 搜到 {len(documents)} 条文档")

    return {"documents": documents}


def grade_documents(state: GraphState):
    """节点：质检"""
    print("---NODE: GRADE (质检中)---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []

    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score

        if grade == "yes":
            print("  - 文档相关 (保留)")
            filtered_docs.append(d)
        else:
            print("  - 文档无关 (剔除)")
            continue

    # 如果只要有一条相关，就认为通过
    global_grade = "yes" if len(filtered_docs) > 0 else "no"
    print(f"  => 最终判定: {global_grade}")

    return {"documents": filtered_docs, "grade": global_grade}


def generate(state: GraphState):
    """节点：生成答案"""
    print("---NODE: GENERATE (生成中)---")
    question = state["question"]
    documents = state["documents"]

    # 把文档拼起来
    context = "\n\n".join([d.page_content for d in documents])

    generation = rag_chain.invoke({"context": context, "question": question})
    return {"generation": generation}


def rewrite_query(state: GraphState):
    """节点：改写问题 (带调试版)"""
    print("---NODE: REWRITE (改写中)---")

    try:
        question = state["question"]

        # 1. 打印一下当前是第几次，确认计数器在工作
        current_step = state.get("loop_step", 0)
        print(f"  [DEBUG] 当前重试次数: {current_step}")

        # 2. 调用改写链
        better_query = question_rewriter.invoke({"question": question})
        print(f"  => 优化后查询: {better_query}")

        # 3. 计数器 +1 并返回
        return {
            "search_query": better_query,
            "loop_step": current_step + 1
        }

    except Exception as e:
        print(f"❌ 改写节点发生严重错误: {e}")
        # 如果出错了，为了不让程序崩溃，我们返回原始问题，并强行增加计数避免死循环
        return {
            "search_query": state["question"],
            "loop_step": state.get("loop_step", 0) + 1
        }

# ===================================================================
# 5. 构建图 (The Graph)
# ===================================================================

workflow = StateGraph(GraphState)

# 添加节点
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("generate", generate)
workflow.add_node("rewrite_query", rewrite_query)

# 定义入口
workflow.set_entry_point("retrieve")

# 添加普通连线
workflow.add_edge("retrieve", "grade_documents")
workflow.add_edge("rewrite_query", "retrieve")
workflow.add_edge("generate", END)


# 定义决策逻辑
def decide_to_generate(state):
    print("---DECISION: 路由判断中---")
    grade = state["grade"]

    # 这里的 .get("loop_step", 0) 是为了防止第一次运行没有这个字段
    current_step = state.get("loop_step", 0)

    # === 熔断逻辑 (Circuit Breaker) ===
    if current_step >= 3:
        print("  => 警告：重试次数已达上限 (3次)！强制结束搜索，尝试硬回答。")
        return "generate"

    if grade == "yes":
        print("  => 决策: 质量达标 -> 生成")
        return "generate"
    else:
        print(f"  => 决策: 质量不达标 (第 {current_step + 1} 次失败) -> 改写")
        return "rewrite_query"

# 添加条件连线
workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "generate": "generate",
        "rewrite_query": "rewrite_query"
    }
)

# 编译
app = workflow.compile()

# ===================================================================
# 6. 测试入口
# ===================================================================
if __name__ == "__main__":
    # 模拟一个模糊的问题，看它会不会触发 Rewrite
    # 比如你的库里有 "量子咖啡机"，但你故意问得模糊一点
    user_input = "那玩意儿大概要预热多久？"

    print(f"\n🚀 开始 Agent 运行测试: {user_input}\n")

    # 必须显式初始化 loop_step 为 0，否则系统不知道从哪开始数
    inputs = {
        "question": user_input,
        "search_query": user_input,
        "loop_step": 0
    }

    # stream_mode="values" 方便调试，看每一步的输出
    for output in app.stream(inputs):
        # 这里的 output 是每一步执行完后的 state 快照
        pass  # 我们在节点内部已经 print 了，所以这里不用打印

    print("\n✅ 流程结束！")