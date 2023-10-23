import sys
import pandas as pd
from sentence_transformers import SentenceTransformer
import argparse
import numpy as np
import gradio as gr
import mdtex2html

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


"""Override Chatbot.postprocess"""


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text

def answer_question(query):
    ret = ""

    # 计算查询字符串与每个问题的相似度
    query_embedding = model.encode([query], normalize_embeddings=True)
    similarities = query_embedding @ model.encode(data["Q"], normalize_embeddings=True).T

    # 找到最相似的问题对应的答案
    max_similarity_index = similarities.argmax()
    if similarities.max() < threshold:
        ret = "作为机器人，我没充分理解你的问题，我给出3个可能性答案供您参考：" + "\n"
        # 获取最相关的前三个答案的索引: 取前3大，之后再倒过来排序使最大的是第一个（以此类推）
        # similarities-> [[0.39948088 0.47452217 0.48097253 0.34293738 0.2999965  0.35529214 ...]]
        max_similarity_indices = np.argsort(similarities[0])[-3:][::-1]
        for index in max_similarity_indices:
            answer = data.loc[index, "A"]
            ret = ret + answer + "\n"
        ret = ret + "如果你认为答案不在其中，请您再次提问：详细描述问题，包括问题背景、问题类型、问题作用、应用场景等等" + "\n"
        return ret
    answer = data.loc[max_similarity_index, "A"]
    ret = ret + answer + "\n"
    return ret

def predict(input, chatbot):
    chatbot.append((parse_text(input), ""))

    chatbot[-1] = (parse_text(input), parse_text(answer_question(input)))
    return chatbot


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], [], None


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">大地保险机器人</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="问题输入...", lines=2).style(
                    container=False)
            #with gr.Column(min_width=32, scale=1):
            with gr.Row():
                submitBtn = gr.Button("提交问题", variant="primary")
                emptyBtn = gr.Button("清空历史")

    history = gr.State([])
    past_key_values = gr.State(None)

    submitBtn.click(predict, [user_input, chatbot],
                    [chatbot], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)

demo.queue().launch(share=True, inbrowser=True)
