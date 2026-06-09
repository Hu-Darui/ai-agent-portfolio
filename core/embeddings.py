# core/embeddings.py
import numpy as np
from openai import OpenAI
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import API_KEY, BASE_URL, EMBEDDING_MODEL

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

def get_embedding(text: str) -> list:
    """获取文本的向量表示"""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a: list, b: list) -> float:
    """计算两个向量的余弦相似度"""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class VectorDB:
    """简易向量数据库"""
    def __init__(self):
        self.documents = []
        self.vectors = []

    def add(self, text: str):
        self.vectors.append(get_embedding(text))
        self.documents.append(text)
        
    def add_batch(self, texts: list):
        """批量添加文档"""
        for text in texts:
            self.add(text)
        print(f"已添加 {len(texts)} 条文档")

    def search(self, query: str, top_k: int = 2) -> list:
        """搜索最相关的文档"""
        query_vec = get_embedding(query)
        scores = [cosine_similarity(query_vec, v) for v in self.vectors]
        ranked = sorted(zip(scores, self.documents), reverse=True)
        return ranked[:top_k]
    
    def load_from_file(self, filepath: str):
        """从文件加载知识库"""
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        self.add_batch(lines)
        return self
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