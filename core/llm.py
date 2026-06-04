# core/llm.py
from openai import OpenAI
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import API_KEY, BASE_URL, CHAT_MODEL

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def call_llm(system: str, user: str, max_retries: int = 3) -> str:
    """带重试机制的 LLM 调用"""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                return f"调用失败：{str(e)}"
            print(f"第{attempt+1}次失败，2秒后重试...")
            time.sleep(2)

def call_llm_stream(system: str, user: str):
    """流式调用，返回生成器"""
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        max_tokens=1000,
        stream=True
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content