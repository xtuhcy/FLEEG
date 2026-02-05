import pandas as pd
import json
from common import MODEL_CALL, MODEL_EEG, SIMI_THRESHOLD
from prompt_template import case2call_prompt, maincall2eeg_prompt, case_prompt, phone_prompt
from models import get_call_response, get_eeg_response
from pathlib import Path
import graph
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm

def gen_police_egg_jsonl(start, end, input_file, output_file):
    # file_path = Path(output_file)
    # file_path.unlink(missing_ok=True)# 文件不存在也不会报错
    # with open(output_file, 'a', encoding='utf-8') as f, open(input_file, 'r', encoding='utf-8') as in_f:
    #     with ThreadPoolExecutor(max_workers=8) as executor:
    #         results = list(tqdm(executor.map(_gen_case_processer, in_f)))
    #     for result in results:
    #         if result != None:
    #             json.dump(result, f, ensure_ascii=False)
    #             f.write('\n')
    file_path = Path(output_file)
    file_path.unlink(missing_ok=True)# 文件不存在也不会报错
    with open(output_file, 'a', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(executor.map(_gen_case_processer, _jsonl_iter(input_file, start, end)), total=(end-start+1)))
        for result in results:
            if result != None:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        
def _gen_case_processer(row):
    try:
        #content = row[1]
        jrow = json.loads(row)
        content = jrow['案情描述']
        code = jrow['案件编号']
        cate = jrow['案件类别']
        #content = re.sub(r'\d+(?=元|张|件|个|台|条|次|万元)', 'x', content)
        call = get_call_response(case_prompt(content), model=MODEL_CALL)
        main_call = "\n".join(line for line in call.splitlines() if not line.startswith("B"))
        eeg = get_eeg_response(maincall2eeg_prompt(main_call), model=MODEL_EEG)
        eeg = eeg.replace("```", "").replace("json", "")
        result = {
            "code": code,
            "cate": cate,
            "content": content,
            "call": call,
            "eeg": eeg
        }
        return result
    except Exception as e:
        print(e)
        return None

# def gen_eeg(input_file = "data/sample.json", output_file = "data/train.jsonl"):
#     file_path = Path(output_file)
#     file_path.unlink(missing_ok=True)# 文件不存在也不会报错
#     with open(output_file, 'a', encoding='utf-8') as f:
#         df = pd.read_json(input_file)
#         with ThreadPoolExecutor(max_workers=8) as executor:
#             results = list(tqdm(executor.map(_gen_processer, df.itertuples(name=None,index=False)), total=len(df)))
#         for result in results:
#             if result != None:
#                 json.dump(result, f, ensure_ascii=False)
#                 f.write('\n')

def _jsonl_iter(input_file, start, end):
    with open(input_file, 'r', encoding='utf-8') as in_f:
        for lineno, line in enumerate(in_f, start=1):
            if lineno >=start and lineno <= end:
                yield line

def gen_eeg_jsonl(start, end, input_file, output_file):
    file_path = Path(output_file)
    file_path.unlink(missing_ok=True)# 文件不存在也不会报错
    with open(output_file, 'a', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(_gen_processer, _jsonl_iter(input_file, start, end)), total=(end-start+1)))
        for result in results:
            if result != None:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')

def _gen_processer(row):
    try:
        #content = row[1]
        jrow = json.loads(row)
        content = jrow['案情描述']
        code = jrow['案件编号']
        cate = jrow['案件类别']
        #content = re.sub(r'\d+(?=元|张|件|个|台|条|次|万元)', 'x', content)
        call = get_call_response(case2call_prompt(content), model=MODEL_CALL)
        if "非电话诈骗" in call:
            print("非电话诈骗")
            return None
        #去除被叫
        main_call = "\n".join(line for line in call.splitlines() if not line.startswith("B"))
        eeg = get_eeg_response(maincall2eeg_prompt(main_call), model=MODEL_EEG)
        eeg = eeg.replace("```", "").replace("json", "")
        result = {
            "code": code,
            "cate": cate,
            "content": content,
            "call": call,
            "eeg": eeg
        }
        return result
    except Exception as e:
        print(e)
        return None

def gen_phone_egg_jsonl(start, end, input_file="data/eval/phone200.xlsx" ,out_file='data/eval/eval_phone_200.jsonl'):
    df_phone = pd.read_excel(input_file)
    json_data = df_phone.to_dict(orient='records')
    with open(out_file, 'w', encoding="utf-8") as f:
        for idx, row in enumerate(json_data):
            if idx >= start and idx <= end:
                content = row['文本']
                call = get_call_response(phone_prompt(content), model=MODEL_CALL)
                main_call = "\n".join(line for line in call.splitlines() if not line.startswith("B"))
                eeg = get_eeg_response(maincall2eeg_prompt(main_call), model=MODEL_EEG)
                result = {
                    "content": content,
                    "call": call,
                    "eeg": eeg
                }
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')

def train(input_file, limit):
    G = None
    count = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for row in f:
            count += 1
            if count % 10 == 0:
                print(f"--------graph has merged : {count}--------")
            data = json.loads(row)
            code = data['code']
            eeg = data["eeg"]
            g = graph.build_nx(code, eeg)
            if g == None:
                continue
            if G == None:
                G = g
            else:
                G = graph.merge(G, g, threshold=SIMI_THRESHOLD)

            if count >= limit:
                break
    return G

def continue_train(input_file, start, end):
    idx = 0
    with open(input_file, 'r', encoding='utf-8') as f:
        for row in f:
            idx += 1
            if idx < start:
                continue
            if idx >= end:
                break
            if idx % 10 == 0:
                print(f"--------graph has merged : {idx}--------")
            data = json.loads(row)
            code = data['code']
            eeg = data["eeg"]
            g = graph.build_nx(code, eeg)
            if g == None:
                continue
            else:
                graph.neo4j_merge(g, threshold=SIMI_THRESHOLD)
            
if __name__ == "__main__":
    #训练数据构造
    gen_eeg_jsonl(start=1001, end=2000, input_file="data/train/sample_5000.jsonl", output_file="data/train/train_1k_2k.jsonl")
    graph.noe4j(train(input_file="data/train.jsonl", limit=2))
    continue_train(input_file="data/train/online.jsonl", start=1, end=60)
    #评估数据：正样本200条
    #gen_eeg_jsonl(start=4801, end=5000, input_file="data/train/sample_5000.jsonl", output_file="data/eval/eval_fraud_200.jsonl")
    #评估数据：负样本100条
    #gen_police_egg_jsonl(start=4701, end=4800, input_file="data/train/sample_5000.jsonl", output_file="data/eval/eval_police_100.jsonl")
    #评估数据：负样本200条
    #gen_phone_egg_jsonl(1, 200)