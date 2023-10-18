import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="BAAI/bge-large-zh-v1.5", help="模型名称")
parser.add_argument("--file_path", default="QA.xlsx", help="文件全路径")
parser.add_argument("--threshold", type=float, default=0.7, help="阈值")
args = parser.parse_args()

# 获取模型名称和文件路径作为启动参数
model_name = args.model_name
file_path = args.file_path
threshold = args.threshold

# 加载模型
model = SentenceTransformer(model_name)

# 加载数据
data = pd.read_excel(file_path)

while True:
    print("")
    print("-----------------------------------------------------------")

    # 提示用户输入查询字符串
    query = input("请输入查询字符串：")

    print("")
    print("-----------------------------------------------------------")

    # 计算查询字符串与每个问题的相似度
    query_embedding = model.encode([query], normalize_embeddings=True)
    similarities = query_embedding @ model.encode(data["Q"], normalize_embeddings=True).T

    # 找到最相似的问题对应的答案
    max_similarity_index = similarities.argmax()
    #print(similarities)
    #print(similarities.max())
    print()
    if similarities.max() < threshold:
        print("作为机器人，我没充分理解你的问题，我给出3个可能性答案供您参考：")
        # 获取最相关的前三个答案的索引: 取前3大，之后再倒过来排序使最大的是第一个（以此类推）
        #similarities-> [[0.39948088 0.47452217 0.48097253 0.34293738 0.2999965  0.35529214 ...]]
        max_similarity_indices = np.argsort(similarities[0])[-3:][::-1]
        #print("max_similarity_indices")
        #print(max_similarity_indices)
        for index in max_similarity_indices:
            answer = data.loc[index, "A"]
            print(answer)
            print("")
        print("如果你认为答案不在其中，请您再次提问：详细描述问题，包括问题背景、问题类型、问题作用、应用场景等等")
        continue
    answer = data.loc[max_similarity_index, "A"]

    # 打印答案
    print("答案：", answer)

#启动程序：python QA.py --model_name BAAI/bge-large-zh-v1.5 --file_path QA.xlsx --threshold 0.7
