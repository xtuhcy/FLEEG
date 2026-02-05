import json
import graph
import pandas as pd

def eval_data_filter():
    """
    过滤train重复数据
    """
    with open("data/eval_300.json", 'r', encoding='utf-8') as f_eval, open("data/sample_2000.json", 'r', encoding='utf-8') as f_train:
        train_data = json.load(f_train)
        eval_data = json.load(f_eval)
        result = []
        for eval in eval_data:
            flag = True
            for train in train_data:
                if(eval['案件编号'] == train['案件编号']):
                    flag = False
                    break
            if flag:
                result.append(eval)
        print(len(result))
        with open('data/eval_200.jsonl', 'w', encoding="utf-8") as outfile:
            for row in result:
                json.dump(row, outfile, ensure_ascii=False)
                outfile.write('\n')

def eval(input_file = "data/eval.jsonl", max_hops=4, key_degree=4, sim_threshold=0.91,match_node_limit=3, match_rate_limit=0.275):
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            code = data['code']
            eeg = data ["eeg"]
            if len(eeg) == 0:
                results.append({
                    "connection": False,
                    "match_count": 0,
                    "match_rate": 0
                })
                continue
            g = graph.build_nx(code, eeg)
            if g == None:
                results.append({
                    "connection": False,
                    "match_count": 0,
                    "match_rate": 0
                })
                continue
            is_match = graph.match(code, g, max_hops=max_hops, key_degree=key_degree, sim_threshold=sim_threshold,match_node_limit=match_node_limit, match_rate_limit=match_rate_limit)
            print(is_match)
            print("--------------------------")
            results.append(is_match)
    return results

if __name__ == "__main__":
    tele_fraud = eval(input_file="data/eval/fraud.jsonl", key_degree=4)
    df_tele_fraud = pd.DataFrame(tele_fraud)
    df_tele_fraud = df_tele_fraud.query("connection == True and (match_count >= 3 and match_count >= 0.5)")
    tp = len(df_tele_fraud)
    tele_phone = eval(input_file="data/eval/normal.jsonl")
    df_tele_phone = pd.DataFrame(tele_phone)
    df_tele_phone = df_tele_phone.query("connection == False or (connection == True and match_count < 3)")
    tn = len(df_tele_phone)

    precision = tp / (tp + 600-tn)
    recall    = tp / (tp + 400-tp)
    f1_manual = 2 * precision * recall / (precision + recall)
    print(f"precision:{precision},recall={recall},f1={f1_manual}")