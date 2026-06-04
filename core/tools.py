# core/tools.py
from langchain_core.tools import tool
from core.embeddings import VectorDB

# 全局向量库实例
db = VectorDB()

def init_knowledge_base(filepath: str):
    """初始化知识库"""
    db.load_from_file(filepath)
    print(f"知识库就绪")

@tool
def search_knowledge(query: str) -> str:
    """搜索公司内部知识库，回答关于公司规定、流程、政策等问题"""
    results = db.search(query, top_k=2)
    if not results:
        return "知识库中没有相关信息"
    return "\n".join([f"- {doc}" for _, doc in results])

@tool
def calculate(expression: str) -> str:
    """计算数学表达式，例如：100 * 0.8、365 / 12"""
    try:
        return str(eval(expression))
    except:
        return "计算失败"

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气情况"""
    data = {
        "北京": "晴天，12°C，北风4级",
        "上海": "多云，18°C，东风2级",
        "台北": "小雨，24°C，南风3级",
    }
    return data.get(city, f"{city}的天气数据暂时无法获取")