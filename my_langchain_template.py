# ========== My LangChain RAG Project Template ==========

# --- 核心组件：模型 (Models) ---

# 1. 大语言模型 (LLM - The Brain)
# 作用：用于最终的思考、推理和生成答案。
# 导入方式：
from langchain.chat_models import init_chat_model

# 2. 嵌入模型 (Embedding Model - The Indexer)
# 作用：将文本转换成语义向量，用于相似度搜索。
# 导入方式：
from langchain_huggingface import HuggingFaceEmbeddings


# --- 核心组件：数据处理 (Data Connection) ---

# 3. 文档加载器 (Loader - The Librarian)
# 作用：从文件（.txt, .pdf, .docx）中读取原始文本。
# 导入方式：
from langchain_community.document_loaders import TextLoader
# from langchain_community.document_loaders import PyPDFLoader # (备用：用于PDF)

# 4. 文本分割器 (Splitter - The Cutter)
# 作用：将长文本切分成适合处理的小块。
# 导入方式：
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 5. 向量数据库 (Vector Store - The Smart File Cabinet)
# 作用：存储文本块和它们的向量，并提供高效的检索功能。
# 导入方式：
from langchain_community.vectorstores import FAISS


# --- 核心组件：编排 (Chains) ---

# 6. 提示模板 (Prompt Template - The Instruction Manual)
# 作用：为 LLM 设计工作指令，告诉它如何利用上下文回答问题。
# 导入方式：
from langchain_core.prompts import ChatPromptTemplate

# 7. 链 (Chains - The Assembly Line)
# 作用：将所有零件组装成自动化的流水线。
# 导入方式：
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# --- 其他辅助工具 ---
import os
from dotenv import load_dotenv
import getpass