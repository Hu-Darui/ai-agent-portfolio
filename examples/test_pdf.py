import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.embeddings import VectorDB

db = VectorDB()
db.load_from_pdf("data/test.pdf")

print(f"\n共加载 {len(db.documents)} 个片段")
print(f"\n前两个片段：")
for doc in db.documents[:2]:
    print(f"  [{len(doc)}字] {doc[:50]}...")

# 测试搜索
query = input("\n输入搜索关键词：")
results = db.search(query, top_k=2)
for score, doc in results:
    print(f"\n相似度 {score:.4f}：\n{doc}")