from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.embeddings import VectorDB
from core.llm import call_llm, call_llm_stream
from fastapi.responses import StreamingResponse

app = FastAPI(title="AI Agent API")

# ========== 启动时初始化知识库 ==========
db = VectorDB()
db.load_from_file("data/knowledge.txt")

# ========== 数据模型 ==========
class QuestionRequest(BaseModel):
    question: str

class ReportRequest(BaseModel):
    topic: str

# ========== 接口1：RAG 问答 ==========
@app.post("/ask")
def ask(request: QuestionRequest):
    results = db.search(request.question, top_k=2)
    context = "\n".join([doc for _, doc in results])
    
    answer = call_llm(
        system="""根据提供的资料回答问题。
如果资料中没有相关信息，说"找不到相关信息"。""",
        user=f"资料：{context}\n\n问题：{request.question}"
    )
    
    return {
        "question": request.question,
        "answer": answer,
        "sources": [doc for _, doc in results]
    }

# ========== 接口2：健康检查 ==========
@app.get("/health")
def health():
    return {
        "status": "ok",
        "knowledge_base_size": len(db.documents)
    }

    # ========== 接口3：流式问答 ==========
@app.post("/ask/stream")
def ask_stream(request: QuestionRequest):
    results = db.search(request.question, top_k=2)
    context = "\n".join([doc for _, doc in results])

    def generate():
        for chunk in call_llm_stream(
            system="根据提供的资料回答问题，没有相关信息就说找不到。",
            user=f"资料：{context}\n\n问题：{request.question}"
        ):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")