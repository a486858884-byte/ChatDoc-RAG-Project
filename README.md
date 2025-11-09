# 智能文档问答系统 (基于 RAG 架构)

这是一个基于 RAG (检索增强生成) 架构的智能问答系统，旨在实现与本地私有文档的自然语言对话。

## 技术栈
- **语言**: Python
- **核心框架**: LangChain
- **模型**: DeepSeek API (LLM), BAAI/bge-large-zh-v1.5 (Embedding)
- **技术**: RAG, FAISS, 语义搜索, 提示工程

## 功能
- 加载本地 .txt 文档作为知识库。
- 对文档进行智能分割、向量化，并使用 FAISS 构建本地索引。
- 实现与文档内容的交互式问答。
- 能够根据文档内容进行重点总结、细节查询。

## 如何运行
1. 克隆本仓库。
2. 在 Conda 中创建并激活环境 `conda create -n langchain_env python=3.11`。
3. 安装所有依赖 `pip install -r requirements.txt`。
4. 创建 `.env` 文件并填入你的 `DEEPSEEK_API_KEY`。
5. 运行 `python chatdoc.py`。