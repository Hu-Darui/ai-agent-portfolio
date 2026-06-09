import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import faiss
import numpy as np
from core.embeddings import get_embedding

class FAISSVectorDB:
    def __init__(self, dimension=1536):
        # 用 HNSW 索引，M=16 表示每个节点最多连接16个邻居
        self.index = faiss.IndexHNSWFlat(dimension, 16)
        self.documents = []
        self.dimension = dimension

    def add(self, text: str):
        vec = get_embedding(text)
        vec_np = np.array([vec], dtype=np.float32)
        self.index.add(vec_np)
        self.documents.append(text)

    def add_batch(self, texts: list):
        for text in texts:
            self.add(text)
        print(f"已添加 {len(texts)} 条文档，使用 HNSW 索引")

    def search(self, query: str, top_k: int = 2) -> list:
        query_vec = np.array([get_embedding(query)], dtype=np.float32)
        # FAISS 返回距离和索引
        distances, indices = self.index.search(query_vec, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:  # -1 表示没找到
                results.append((float(1 - dist), self.documents[idx]))
        return results

    def load_from_file(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        self.add_batch(lines)
        return self

# 在文件底部加测试代码
if __name__ == "__main__":
    from core.embeddings import VectorDB
    
    texts = [
        "员工请假需要提前3天在OA系统申请，直属上级审批后生效。",
        "公司报销流程：填写报销单，附发票，提交财务部，5个工作日内到账。",
        "试用期为3个月，转正需要通过技术考核和主管评估。",
        "年假根据工龄计算：1-3年5天，3-5年10天，5年以上15天。",
        "公司每周五下午4点举行全员例会，地点在三楼会议室。",
    ]
    
    query = "我想请假怎么办"
    
    # 原始暴力搜索
    print("=== SimpleVectorDB ===")
    simple_db = VectorDB()
    simple_db.add_batch(texts)
    for score, doc in simple_db.search(query):
        print(f"  {score:.4f}: {doc[:30]}")
    
    # FAISS HNSW 搜索
    print("\n=== FAISSVectorDB ===")
    faiss_db = FAISSVectorDB()
    faiss_db.add_batch(texts)
    for score, doc in faiss_db.search(query):
        print(f"  {score:.4f}: {doc[:30]}")
def load_from_pdf(self, filepath: str, chunk_size: int = 300):
    """从 PDF 加载知识库，自动切片"""
    import fitz
    
    doc = fitz.open(filepath)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    
    # 按段落切片
    chunks = self._chunk_text(full_text, chunk_size)
    self.add_batch(chunks)
    print(f"从 PDF 加载了 {len(chunks)} 个片段")
    return self

def _chunk_text(self, text: str, chunk_size: int = 300) -> list:
    """递归切片"""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    paragraphs = text.split("\n\n")
    
    if len(paragraphs) > 1:
        current = ""
        for para in paragraphs:
            if len(current) + len(para) <= chunk_size:
                current += para + "\n\n"
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = para + "\n\n"
        if current.strip():
            chunks.append(current.strip())
        return chunks
    
    # 段落切不了，按句子切
    sentences = text.split("。")
    current = ""
    for s in sentences:
        if len(current) + len(s) <= chunk_size:
            current += s + "。"
        else:
            if current.strip():
                chunks.append(current.strip())
            current = s + "。"
    if current.strip():
        chunks.append(current.strip())
    return chunks