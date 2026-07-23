import torch
import random
import heapq
import math
from tqdm import tqdm
import numpy as np
import argparse
from train import (model_args,
                   cnn_config,
                   transformer_config,
                   gnn_config,
                   INPUT_DIM, OUTPUT_DIM,NUM_LAYERS,L,PATH,M,HIDDEN_SIZE,DROP_OUT,DO_STD)
from model import MyClassifier

# 设置容忍度
parser = argparse.ArgumentParser()
parser.add_argument("--tolerance", type=float, default=5e-2)
args = parser.parse_args()
Tolerance = args.tolerance
print(f"Using tolerance: {Tolerance}")

"""abbanden correspondence of indexes, use global random generator"""
"""mluti-ans is considered"""
IT_rng = np.random.default_rng(int(1e19))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dijkstra(adj, start, end):
    """Dijkstra 算法，返回 (最短距离, 路径节点列表)"""
    n = len(adj)
    dist = [float('inf')] * n
    prev = [-1] * n
    dist[start] = 0
    pq = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == end:
            break
        for v in range(n):
            if adj[u][v] > 0:
                nd = d + adj[u][v]
                if nd < dist[v]:
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(pq, (nd, v))

    if dist[end] == float('inf'):
        return None, None
    
    path = []
    cur = end
    while cur != -1:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return dist[end], path

if __name__ == "__main__":
    # ---------- 加载模型 ----------
    model = MyClassifier(**model_args, **cnn_config, **transformer_config,**gnn_config).to(DEVICE)
    model.load_state_dict(torch.load(PATH, map_location=DEVICE))
    model.eval()
    print("模型权重加载成功")

    inputs,labels=torch.load("eval_data/inputs.pt"),torch.load("eval_data/labels.pt")

    # ---------- 推理 ----------
    TEST_LEN = 10000

    model.eval()
    right=0
    origin_right=0

    with torch.no_grad():
        for i in tqdm(range(TEST_LEN),mininterval=1):
            batch = inputs[i].to(DEVICE)
            ans = labels[i].item()
            adj = batch[0, :100].reshape(10, 10).tolist()
            start, end = map(int,batch[0, -2:].cpu().tolist())
            Min,_=dijkstra(adj,start,end)
            batch=batch.to(DEVICE)
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).cpu().item()
            if preds == ans:
                right+=1
                origin_right+=1
            else:
                if preds in [start,end]:
                    continue
                new_length= dijkstra(adj,preds,end)[0]
                if new_length==None:
                    continue
                ans_length =  new_length +  adj[start][preds]
                assert ans_length >= Min, "fix me"
                if abs(ans_length-Min) <= Tolerance:
                    right+=1


    acc =right / TEST_LEN
    origin_acc=origin_right / TEST_LEN
    print(f"acc: {acc:.4f},origin_acc: {origin_acc:.4f}")
