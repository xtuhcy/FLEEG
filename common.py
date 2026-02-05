import ast

SIMI_THRESHOLD=0.85
LLM_API_CALL="https://api.deepseek.com/v1"
LLM_API_CALL_KEY="DEEPSEEK_API_KEY"
LLM_API_EEG="https://api.deepseek.com/v1"
LLM_API_EEG_KEY="DEEPSEEK_API_KEY"
EMBEDDING_API="https://dashscope.aliyuncs.com/compatible-mode/v1"
EMBEDDING_API_KEY="DASHSCOPE_API_KEY"

MODEL_CALL = "deepseek-chat"
MODEL_EEG = "deepseek-chat"
MODEL_EMBEDDING = "text-embedding-v4"
DIM_EMBEDDING=512
NEO_URI = "bolt://localhost:7687"
NEO_USER = "neo4j"
NEO_PWD = "xxx"
NEO_DB = "fraud8k"
NEO_MATCH_K = 1

def event2tuple(event):
    def format(event):
        return event.replace("'", "").replace("(", "('").replace(")", "')").replace(",", "','")
    return ast.literal_eval(format(event))
    