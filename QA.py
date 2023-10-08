import sys
import pandas as pd
from sentence_transformers import SentenceTransformer

# 获取模型名称和文件路径作为启动参数
model_name = "BAAI/bge-large-zh-v1.5"
file_path = "QA.xlsx"

if len(sys.argv) > 1:
    model_name = sys.argv[1]
if len(sys.argv) > 2:
    file_path = sys.argv[2]

# 加载模型
model = SentenceTransformer(model_name)

# 加载数据
data = pd.read_excel(file_path)

while True:
    # 提示用户输入查询字符串
    query = input("请输入查询字符串：")

    # 计算查询字符串与每个问题的相似度
    query_embedding = model.encode([query], normalize_embeddings=True)
    similarities = query_embedding @ model.encode(data["Q"], normalize_embeddings=True).T

    # 找到最相似的问题对应的答案
    max_similarity_index = similarities.argmax()
    if max_similarity_index < 0.5:
        print("不好意思，没有合适的答案")
        continue
    answer = data.loc[max_similarity_index, "A"]

    # 打印答案
    print(similarities)
    print("答案：", answer)

#启动程序：python QA.py BAAI/bge-large-zh-v1.5 QA.xlsx