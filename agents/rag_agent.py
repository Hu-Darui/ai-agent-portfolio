# agents/rag_agent.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from core.tools import search_knowledge, calculate, get_weather, init_knowledge_base
from config import API_KEY, BASE_URL, CHAT_MODEL

SYSTEM_PROMPT = """你是一个公司内部助手。
回答公司相关问题时，必须先使用 search_knowledge 工具查询知识库。
遇到计算问题时，使用 calculate 工具。
查询天气时，使用 get_weather 工具。
其他问题直接回答。"""

def create_rag_agent():
    llm = ChatOpenAI(
        base_url=BASE_URL,
        api_key=API_KEY,
        model=CHAT_MODEL,
        max_tokens=1000
    )
    return create_react_agent(
        model=llm,
        tools=[search_knowledge, calculate, get_weather],
        prompt=SYSTEM_PROMPT
    )

def run_rag_agent():
    init_knowledge_base("data/knowledge.txt")
    agent = create_rag_agent()
    chat_history = []

    print("RAG Agent 就绪（输入 exit 退出）\n")
    while True:
        user_input = input("你：").strip()
        if user_input == "exit":
            break
        if not user_input:
            continue

        chat_history.append(HumanMessage(content=user_input))
        result = agent.invoke({"messages": chat_history})
        reply = result["messages"][-1].content
        chat_history.append(result["messages"][-1])
        print(f"AI：{reply}\n")

if __name__ == "__main__":
    run_rag_agent()