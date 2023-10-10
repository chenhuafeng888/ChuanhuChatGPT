import sys
import pandas as pd
from transformers import AutoTokenizer, AutoModel

# 获取模型名称和文件路径作为启动参数
model_name = "THUDM/chatglm2-6b-32k-int4"
file_path = "恶性肿瘤保险条款.txt"

if len(sys.argv) > 1:
    model_name = sys.argv[1]
if len(sys.argv) > 2:
    file_path = sys.argv[2]

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
model = model.eval()
print("模型已加载：" + model_name)

# 加载数据
with open(file_path, 'r', encoding='utf-8') as file:
    document_text = file.read()
#print(document_text)
#print(len(document_text))

while True:
    # 提示用户输入查询字符串
    query = input("请输入查询字符串：")

    #使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要尝试编造答案。 Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    #你作为有经验的保险顾问，请基于上面内容进行充分理解并回答用户问题，问：我现在已经95岁了，能不能投保？答：
    response, history = model.chat(tokenizer, document_text + "。\n" + query, history=[])
    # response, history = model.chat(tokenizer, document_text + "\n请根据上面内容，" + query, history=[])
    # response, history = model.chat(tokenizer, document_text + "\n你作为聊天机器人，只能根据上面内容回复，不要有大模型幻想的内容，也不要基于你自己的知识产生内容，如果上面内容里找不到答案你就说：您的问题超出知识库范围。" + query, history=[])

    print(response)
    print("#######################################\n")

#!pip install protobuf transformers==4.30.2 cpm_kernels torch>=2.0 gradio mdtex2html sentencepiece accelerate
#启动程序：python robot.py THUDM/chatglm2-6b-32k-int4 恶性肿瘤保险条款.txt
