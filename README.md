# 🤖 ChatDoc: Enterprise-Grade RAG Knowledge Base
![Image](https://github.com/user-attachments/assets/145a3332-3d32-479f-8412-b426e1f2bb4c)

## 📖 Introduction
ChatDoc 是一个基于 **RAG (Retrieval-Augmented Generation)** 架构的垂直领域智能问答系统。它解决了 LLM 的知识滞后与幻觉问题，实现了从文档解析、向量检索到持久化存储的全链路闭环。

## 🚀 Key Features
- **RAG 核心**: 基于 LangChain + FAISS 实现动态文本分块与语义检索。
- **大模型驱动**: 集成 DeepSeek API (OpenAI Compatible)，引入思维链 (CoT) 降低幻觉。
- **持久化记忆**: 采用 MySQL 存储全量历史对话，支持审计与回溯。
- **可视化交互**: 基于 Streamlit 构建的现代化 Chat UI，支持流式响应。

## 🛠️ Tech Stack
- **Framework**: Python 3.9+, LangChain
- **Model**: DeepSeek-V3, BGE-Large-Zh (Embedding)
- **Database**: FAISS (Vector), MySQL (Relational)
- **Interface**: Streamlit

## ⚡ Quick Start

1. Clone Repo
git clone https://github.com/a486858884-byte/ChatDoc-RAG-Project.git
cd ChatDoc-RAG-Project
2. Install Dependencies
pip install -r requirements.txt
3. Configure Environment
Create a .env file:DEEPSEEK_API_KEY=sk-xxxxxx
4. Run App
streamlit run app.py