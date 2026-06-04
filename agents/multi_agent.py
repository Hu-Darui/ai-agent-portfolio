# agents/multi_agent.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from concurrent.futures import ThreadPoolExecutor
from core.llm import call_llm

def planner_agent(topic: str) -> str:
    print("\n[Planner] 制定研究提纲...")
    result = call_llm(
        system="""你是一个研究规划专家。
输出3-4个核心研究问题，每个问题一行，用数字编号。
只输出提纲，不要其他内容。""",
        user=f"话题：{topic}"
    )
    print(f"提纲：\n{result}")
    return result

def researcher_agent_pros(topic: str, outline: str) -> str:
    print("\n[Researcher A] 研究优点...")
    return call_llm(
        system="你是研究员，专注分析给定话题的优点和机会，列出3-5个要点。",
        user=f"话题：{topic}\n提纲：{outline}"
    )

def researcher_agent_cons(topic: str, outline: str) -> str:
    print("\n[Researcher B] 研究缺点...")
    return call_llm(
        system="你是研究员，专注分析给定话题的缺点和挑战，列出3-5个要点。",
        user=f"话题：{topic}\n提纲：{outline}"
    )

def writer_agent(topic: str, research: str) -> str:
    print("\n[Writer] 撰写最终报告...")
    return call_llm(
        system="""你是专业报告撰写专家。
根据研究要点写一篇结构完整的分析报告。
要求：有标题、引言、章节小标题、总结，500字左右。""",
        user=f"话题：{topic}\n\n研究要点：\n{research}"
    )

def run_pipeline(topic: str) -> str:
    print(f"\n{'='*50}")
    print(f"开始处理：{topic}")
    print(f"{'='*50}")

    outline = planner_agent(topic)

    print("\n[并行] 两个 Researcher 同时工作...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_pros = executor.submit(researcher_agent_pros, topic, outline)
        future_cons = executor.submit(researcher_agent_cons, topic, outline)
        pros = future_pros.result()
        cons = future_cons.result()

    research = f"【优点】\n{pros}\n\n【缺点】\n{cons}"
    return writer_agent(topic, research)

if __name__ == "__main__":
    topic = input("请输入研究话题：").strip()
    report = run_pipeline(topic)

    print(f"\n{'='*50}")
    print("最终报告")
    print(f"{'='*50}")
    print(report)

    with open("report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("\n报告已保存到 report.txt")