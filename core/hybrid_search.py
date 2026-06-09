import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import jieba
from rank_bm25 import BM25Okapi
from core.embeddings import VectorDB

def tokenize(text: str) -> list:
    """中文分词"""
    return list(jieba.cut(text))

class HybridSearchDB:
    def __init__(self):
        self.vector_db = VectorDB()
        self.documents = []
        self.bm25 = None

    def add_batch(self, texts: list):
        """批量添加文档"""
        self.documents = texts
        
        # 向量库
        self.vector_db.add_batch(texts)
        
        # BM25 索引（需要分词）
        tokenized = [tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        print(f"混合索引建立完成，共 {len(texts)} 条文档")

    def vector_search(self, query: str, top_k: int = 5) -> list:
        """向量检索"""
        results = self.vector_db.search(query, top_k=top_k)
        return [(doc, score) for score, doc in results]

    def bm25_search(self, query: str, top_k: int = 5) -> list:
        """BM25 关键词检索"""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(
            zip(scores, self.documents),
            reverse=True
        )[:top_k]
        return [(doc, score) for score, doc in ranked]

    def rrf_fusion(self, results_list: list, k: int = 60) -> list:
        """
        RRF 融合排序（Reciprocal Rank Fusion）
        把多个排序列表融合成一个
        公式：score = 1 / (k + rank)
        """
        doc_scores = {}
        
        for results in results_list:
            for rank, (doc, _) in enumerate(results):
                if doc not in doc_scores:
                    doc_scores[doc] = 0
                doc_scores[doc] += 1 / (k + rank + 1)
        
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked

    def search(self, query: str, top_k: int = 3) -> list:
        """混合检索主入口"""
        # 两种方式各取5个结果
        vector_results = self.vector_search(query, top_k=5)
        bm25_results   = self.bm25_search(query, top_k=5)
        
        # RRF 融合
        fused = self.rrf_fusion([vector_results, bm25_results])
        
        return fused[:top_k]


if __name__ == "__main__":
    # 安装分词库
    # pip install jieba rank-bm25

    texts = [
        "员工请假需要提前3天在OA系统申请，直属上级审批后生效。",
        "公司报销流程：填写报销单，附发票，提交财务部，5个工作日内到账。",
        "试用期为3个月，转正需要通过技术考核和主管评估。",
        "年假根据工龄计算：1-3年5天，3-5年10天，5年以上15天。",
        "公司OA系统网址是 http://oa.company.com，用工号登录。",
        "公司每周五下午4点举行全员例会，地点在三楼会议室。",
    ]

    db = HybridSearchDB()
    db.add_batch(texts)

    # 测试1：语义查询（向量检索擅长）
    print("\n=== 测试1：语义查询 ===")
    query1 = "我想歇几天"
    print(f"查询：{query1}")
    for doc, score in db.search(query1):
        print(f"  {score:.4f}: {doc[:40]}")

    # 测试2：精确查询（BM25 擅长）
    print("\n=== 测试2：精确查询 ===")
    query2 = "OA系统网址"
    print(f"查询：{query2}")
    for doc, score in db.search(query2):
        print(f"  {score:.4f}: {doc[:40]}")