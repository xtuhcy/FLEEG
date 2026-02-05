from typing import List
import networkx as nx
import numpy as np
import json
import ast
import re
from openai.types.embedding import Embedding
from common import *
from models import get_embedding
import traceback
from neo4j import GraphDatabase, Result
import time
import atexit
from node_embeddings import Node_Embeddings

driver = GraphDatabase.driver(NEO_URI, auth=(NEO_USER, NEO_PWD), database=NEO_DB)
# 注册退出钩子
atexit.register(driver.close)

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b) if norm_a * norm_b != 0 else 0  # 避免除零错误

def merge(G1, G2, threshold):
    merged = nx.compose(G1, G2)
    for node1 in G1.nodes():
        for node2 in G2.nodes():
            if node1 == node2:
                #如果node1和node2完全一样，已经在compose阶段合并完成
                print(f"has compose : {node1} - {node2}")
                continue
            try:
                if node2 in merged:
                    node1_embedding = G1.nodes[node1]['embedding']
                    node2_embedding = G2.nodes[node2]['embedding']
                    cs = cosine_similarity(node1_embedding, node2_embedding)
                    if cs > threshold:
                        print(f"{node1}-{node2}-{cs}")
                        nx.contracted_nodes(merged, node1, node2, copy=False, store_contraction_as=None)  # 合并相似节点
                        #对特征向量进行mean
                        merged.nodes[node1]['embedding'] = [(x + y) / 2 for x, y in zip(node1_embedding, node2_embedding)]
                #else:
                #    print(f"has merged : {node1} - {node2}")
            except Exception as ex:
                print(f"contracted_nodes error : {ex}")
                traceback.print_exc()
                print(f"G1:{node1}:{G1.nodes[node1]}-G2:{node2}:{G2.nodes[node2]}")
                print("-"*20)
    return merged

def neo4j_merge(G, threshold):
     #待识别图节点列表
    nodes = list(map(str, G.nodes()))
    print(nodes)
    #待识别节点列表向量化
    query_vec: List[Embedding] = _object_embedding(None, nodes)
    #节点向量匹配
    with driver.session() as s:
        result: Result = s.run("""
            WITH $query_vectors AS vectors
            UNWIND vectors AS q_vec
            CALL db.index.vector.queryNodes('fraud_embedding_idx', $k, q_vec)
            YIELD node, score
            RETURN node.id AS id, node.embedding as embedding, score
        """, k=NEO_MATCH_K, query_vectors=query_vec)
        #向量匹配的所有节点
        matchs = [(r["id"], r["embedding"], r["score"]) for r in result]
        print([(r[0], r[2]) for r in matchs])
        for i in range(len(nodes)-1):
            #左节点
            u = (nodes[i], query_vec[i])
            u_x = matchs[i]
            if u_x[2] > threshold:
                embedding = [(x + y) / 2 for x, y in zip(u_x[1], u[1])]
                s.run("MATCH (n:Fraud {id: $id}) SET n.embedding = $embedding RETURN n",id=u_x[0], embedding=embedding)
            else:
                s.run("MERGE (n:Fraud {id: $id}) ON CREATE SET n.embedding = $embedding", id=u[0], embedding=u[1])
                u_x = u
            #右节点
            v = (nodes[i+1], query_vec[i+1])
            v_x = matchs[i+1]
            if v_x[2] > threshold:
                embedding = [(x + y) / 2 for x, y in zip(v_x[1], v[1])]
                s.run("MATCH (n:Fraud {id: $id}) SET n.embedding = $embedding RETURN n",id=v_x[0], embedding=embedding)
            else:
                s.run("MERGE (n:Fraud {id: $id}) ON CREATE SET n.embedding = $embedding", id=v[0], embedding=v[1])
                v_x = v
            #边
            print(f"add edge : {u_x[0]} -NEXT-> {v_x[0]}")
            s.run("""
                MATCH (a:Fraud {id:$u}), (b:Fraud {id:$v})
                MERGE (a)-[r:NEXT]->(b)
            """, u=u_x[0], v=v_x[0])
    
def _node_vector_match(query_vec, threshold=0.91):
    with driver.session() as s:
        result: Result = s.run("""
            WITH $query_vectors AS vectors
            UNWIND vectors AS q_vec
            CALL db.index.vector.queryNodes('fraud_embedding_idx', $k, q_vec)
            YIELD node, score
            RETURN node.id AS id, score
        """, k=NEO_MATCH_K, query_vectors=query_vec)
        #向量匹配的所有节点
        matchs = [(r["id"], r["score"]) for r in result]
        print([row[1] for row in matchs])
        #相似度大于threshold的所有节点
        ids = [row[0] for row in matchs if row[1] >= threshold]
        print(ids)
        return ids

def build_nx(code, eeg):
    """
    {
        "nodes": {
            "1": "诈骗者假冒京东金融客服，声称金条功能异常激活影响征信",
            "2": "受害者信以为真，表达担忧",
            "3": "诈骗者要求操作京东APP取消异常",
            "4": "受害者同意操作",
            "5": "诈骗者指导打开京东APP进入金条页面"
        },
        "edges": [
            {
                "from": "1",
                "to": "2"
            },
            {
                "from": "2",
                "to": "3"
            },
            {
                "from": "3",
                "to": "4"
            },
            {
                "from": "4",
                "to": "5"
            }
        ]
    }
    """
    G = nx.DiGraph()
    #print(eeg)
    eeg = re.sub(r'\d+(?=元|张|件|个|台|条|次|万元)', 'x', eeg)
    try:
        events = json.loads(eeg.replace("```", "").replace("json", ""))
        #print(events)
        embeddings = _object_embedding(code, events)
        if embeddings == None or len(embeddings) == 0:
            return None
        limit = len(embeddings)
        for idx, event in enumerate(events):
            if idx < limit:
                G.add_node(event, embedding=embeddings[idx])
        for idx in range(len(events) - 1):
            if idx < limit - 1:
                e_from = events[idx]
                e_to = events[idx+1]
                G.add_edge(e_from, e_to)
        return G
    except Exception as e:
        print(f"ex:{e},eeg:{eeg}")
        return None

def noe4j(G):
    # 1. 连接
    with driver.session() as s:
        # 2. 创建embedding索引
        s.run("DROP INDEX fraud_embedding_idx IF EXISTS")
        s.run(f"""
            CREATE VECTOR INDEX fraud_embedding_idx IF NOT EXISTS
            FOR (n:Fraud)
            ON (n.embedding)
            OPTIONS {{
            indexConfig: {{
                `vector.dimensions`: {DIM_EMBEDDING},
                `vector.similarity_function`: 'cosine'
            }}
            }}
        """)
        # 3. 把 NetworkX 图写成 Cypher
        # 先删旧图（可选）
        s.run("MATCH (n) DETACH DELETE n")
        # 写节点
        for n, d in G.nodes(data=True):
            s.run("CREATE (n:Fraud {id:$id, embedding:$embedding})", id=n, embedding=d['embedding'])
        # 写边
        for u, v in G.edges(data=False):
            s.run("""
                MATCH (a:Fraud {id:$u}), (b:Fraud {id:$v})
                CREATE (a)-[r:NEXT]->(b)
            """, u=u, v=v)

def _object_embedding(code, events):
    try:
        if code != None and code in Node_Embeddings:
            return Node_Embeddings[code]
        else:
            embeddings = [e.embedding for e in get_embedding(events, model=MODEL_EMBEDDING, dim=DIM_EMBEDDING)]
            if code != None:
                Node_Embeddings[code] = embeddings
            return embeddings
    except Exception as e:
        print(f"node embedding error : {e}|events : {events}")
        return None

def match(code, G, max_hops=4, key_degree=4, sim_threshold=0.91, match_node_limit=3, match_rate_limit=0.275):
    #待识别图节点列表
    nodes = list(map(str, G.nodes()))
    print(nodes)
    #待识别节点列表向量化
    query_vec: List[Embedding] = _object_embedding(code, nodes)
    #节点向量匹配
    ids = _node_vector_match(query_vec,threshold=sim_threshold)
    if len(ids) == 0:
        return {
            "nodes": nodes,
            "key_nodes": [],
            "connection": False,
            "match_count": len(ids),
            "match_rate": 0
        }
    #关键节点匹配
    match_nodes = _key_node_match(ids, key_degree=key_degree)
    #节点匹配度
    match_rate = len(match_nodes) / len(nodes)
    #print(f"match nodes:{match_nodes}. match rate: {match_rate*100}%")
    #是否联通
    connection = _is_all_conn(match_nodes, max_hops)
    return {
        "nodes": nodes,
        "key_nodes": match_nodes,
        "connection": connection,
        "match_count": len(match_nodes),
        "match_rate": match_rate
    }
    # #匹配的节点数大于等于x，匹配率高于x
    # if len(match_nodes) >= match_node_limit and match_rate > match_rate_limit:
    #     #查看匹配节点的连通性
    #     return 
    # else:
    #     return False

def _node_vector_match(query_vec, threshold=0.91):
    with driver.session() as s:
        result: Result = s.run("""
            WITH $query_vectors AS vectors
            UNWIND vectors AS q_vec
            CALL db.index.vector.queryNodes('fraud_embedding_idx', $k, q_vec)
            YIELD node, score
            RETURN node.id AS id, score
        """, k=NEO_MATCH_K, query_vectors=query_vec)
        #向量匹配的所有节点
        matchs = [(r["id"], r["score"]) for r in result]
        print([row[1] for row in matchs if row[1]])
        #相似度大于threshold的所有节点
        ids = [row[0] for row in matchs if row[1] >= threshold]
        print(ids)
        return ids

def _key_node_match(ids, key_degree=5):
     #匹配的各个节点的出入度和
    degrees = _degree(ids)
    #print(degrees)
    #过滤掉小于9的非重要节点
    nodes = [l for l, i in zip(ids, degrees) if i >= key_degree]
    return nodes

def _build_query(sequence, max_hops=4):
    parts = []
    for i in range(len(sequence)-1):
        q = f"""
        EXISTS {{
          MATCH (a:Fraud {{id: '{sequence[i]}'}})-[:NEXT*1..{max_hops}]->(b:Fraud {{id: '{sequence[i+1]}'}})
        }}"""
        parts.append(q)
    return "RETURN " + " AND ".join(parts) + " AS path_exists"

def _is_all_conn(ids: List[str], max_hops) -> bool:
    start = time.perf_counter()
    if len(ids) <= 1:
        return False
    with driver.session() as s:
        query = _build_query(ids, max_hops=max_hops)
        #print(query)
        result = s.run(query).value()
        #print(f"connection spend time : {time.perf_counter() - start:.6f} 秒")
        return result[0]
    
def _degree(seq):
    degrees = []
    with driver.session() as s:
        for node in seq:
            result = s.run("""
                        MATCH (n:Fraud {id: $nd})
                        RETURN
                            COUNT { MATCH (n)-[:NEXT]->() } +
                            COUNT { MATCH (n)<-[:NEXT]-() } AS total_rel_degree;
                       """, nd=node).value()
            if len(result) > 0:
                degrees.append(result[0])
            else:
                degrees.append(0)
    return degrees

if __name__ == "__main__":
    eeg = """
    {
        "nodes": {
            "1": "(诈骗者,自称,快手官方客服)",
            "2": "(诈骗者,声称,用户开通了黄金会员服务并会扣费)",
            "3": "(诈骗者,提出,取消会员服务)",
            "4": "(诈骗者,要求,归集银行卡资金到一张卡上)",
            "5": "(诈骗者,要求,转账20000元作为验证资金)"
        },
        "edges": [
            {
                "from": "1",
                "to": "2"
            },
            {
                "from": "2",
                "to": "3"
            },
            {
                "from": "3",
                "to": "4"
            },
            {
                "from": "4",
                "to": "5"
            }
        ]
    }
    """
    #print(ast.literal_eval("('诈骗者','自称','快手官方客服')"))
    #print(build_nx(eeg))
    #seq = ['自称快手官方客服', '通知产品召回', '要求开通支付宝备用金', '索要银行登录密码和验证码', '要求下载远程协助工具', '索要银行登录密码和验证码', '要求购买电子礼品卡']
    #seq = ['确认受害人位置']
    seq = ['自称旗舰店客服', '说明高利率问题', '提出办理降息业务', '指示注册账号', '指示加入安全会议', '要求提供信息', '指导操作转账流程', '指导发起多笔验证转账', '承诺处理退款事宜']
    degrees = _degree(seq)
    print(degrees)
    print([l for l, i in zip(seq, degrees) if i >= 9])
    labels_new = [l for l, i in zip(seq, degrees) if i >= 9]
    print(labels_new)
    query = _build_query(labels_new, max_hops=3)
    print(query)
    #query="MATCH (n0:Fraud {id: '声称账户产生不良记录'})-[:REL*1..3]->(n1:Fraud {id: '要求下载钉钉软件'})-[:REL*1..3]->(n2:Fraud {id: '要求集中资金'})-[:REL*1..3]->(n3:Fraud {id: '提供银行账户信息'}) RETURN 1 LIMIT 1;"
    with driver.session() as s:
        result = s.run(query).value()
        print(result[0])