import os
from typing import List
from openai.types.create_embedding_response import CreateEmbeddingResponse
from openai.types.embedding import Embedding
from common import *
from config.load_key import load_key
from openai import OpenAI
import os

load_key()
# client = OpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

client_llm_call = OpenAI(
    base_url=LLM_API_CALL,
    api_key=os.getenv(LLM_API_CALL_KEY), 
)

client_llm_eeg = OpenAI(
    base_url=LLM_API_EEG,
    api_key=os.getenv(LLM_API_EEG_KEY), 
)

client_embedding = OpenAI(
    api_key=os.getenv(EMBEDDING_API_KEY), 
    base_url=EMBEDDING_API,
)

def get_call_response(prompt, model):
    response = client_llm_call.chat.completions.create(
        model=model,
        messages=[
            #{"role": "system", "content": "你是一个反电话诈骗的专家"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        stream=False
    )
    return response.choices[0].message.content
    
def get_eeg_response(prompt, model):
    response = client_llm_eeg.chat.completions.create(
        model=model,
        messages=[
            #{"role": "system", "content": "你是一个反电话诈骗的专家"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
        top_p=0.1,
        stream=False
    )
    return response.choices[0].message.content


def get_embedding(inputs, model="text-embedding-v4", dim=512) -> List[Embedding]:
    completion: CreateEmbeddingResponse = client_embedding.embeddings.create(
        model=model,
        input=inputs[:10],#不能超过10个
        dimensions=dim, # 指定向量维度（仅 text-embedding-v3及 text-embedding-v4支持该参数）
        encoding_format="float"
    )
    return completion.data

if __name__ == "__main__":
    embeddings: List[Embedding] = get_embedding(["你好"])
    for row in embeddings:
        print(row.embedding)

if __name__ == "__main__":
    from prompt_template import case2call_prompt, call2eeg_prompt
    content="""
A: 您好，这里是XX市公安局刑侦大队，我是王警官，警号XXXXXX。请问是B女士吗？
B: 啊？是的，我是。请问有什么事吗？
A: 我们刚破获一起重大洗钱案件，发现您的身份证信息被犯罪分子冒用开设了多个银行账户，涉嫌参与洗黑钱活动。现在需要您配合调查。
B: 什么？这不可能！我从来没有做过违法的事啊！
A: 请您保持冷静。我们理解您的担忧，但现在系统显示确实有您的涉案记录。为了证明您的清白，我们需要您配合进行资金核查。您现在方便加一下我们专案组的微信吗？
B: 好的好的，我该怎么加？
A: 您记一下，微信号是XXXXXX，昵称是"全"。添加时请备注"配合调查"，我会把相关案件材料发给您核实。
B: 我马上加，请一定要帮我查清楚！
A: 请您放心，我们公安机关一定会秉公办理。但请注意，此案涉及国家机密，在调查期间请不要向任何人透露相关信息，否则会影响案件侦办。
"""
    call = get_eeg_response(call2eeg_prompt(content), model=MODEL_EEG)
    print(call)