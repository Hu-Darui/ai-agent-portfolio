markdown# AI Agent Portfolio

一个从零实现的 AI Agent 系统，包含 RAG 知识库问答、多 Agent 协作流水线和自动评估系统。

## 项目结构
ai-agent-portfolio/
├── core/              # 核心模块
│   ├── llm.py         # LLM 调用封装（含重试机制）
│   ├── embeddings.py  # Embedding 和向量数据库
│   └── tools.py       # Agent 工具定义
├── agents/            # Agent 实现
│   ├── rag_agent.py   # RAG 知识库问答 Agent
│   ├── multi_agent.py # 多 Agent 并行流水线
│   └── evaluator.py   # 自动评估系统
└── data/              # 知识库数据

## 功能特性

- **RAG 问答 Agent**：基于向量检索的知识库问答，支持工具调用（天气查询、数学计算）
- **多 Agent 流水线**：Planner → Researcher（并行）→ Writer 三阶段协作生成报告
- **自动评估系统**：关键词匹配 + LLM-as-Judge 双层评估，量化 Agent 回答质量

## 技术栈

- **LLM 调用**：OpenAI SDK + OpenRouter
- **向量检索**：text-embedding-3-small + 余弦相似度
- **Agent 框架**：LangGraph
- **并行执行**：Python ThreadPoolExecutor

## 快速开始

1. 安装依赖

```bash
pip install openai numpy langgraph langchain-openai
```

2. 配置 API Key

```python
# 复制 config.example.py 为 config.py，填入你的 Key
cp config.example.py config.py
```

3. 运行 RAG Agent

```bash
python agents/rag_agent.py
```

4. 运行多 Agent 报告生成

```bash
python agents/multi_agent.py
```

5. 运行评估系统

```bash
python agents/evaluator.py
```

## 核心实现亮点

### RAG 检索流程
用户问题 → Embedding 向量化 → 余弦相似度检索 → 相关文档注入 Prompt → LLM 生成回答

### Agent Loop
模型调用 → 检查 tool_calls → 执行工具 → 结果加入历史 → 循环直到无工具调用

### 并行多 Agent
```python
with ThreadPoolExecutor(max_workers=2) as executor:
    future_pros = executor.submit(researcher_agent_pros, topic, outline)
    future_cons = executor.submit(researcher_agent_cons, topic, outline)
```

## 评估结果

| 指标 | 结果 |
|------|------|
| 关键词通过率 | 100% |
| LLM 平均评分 | 4.87/5 |