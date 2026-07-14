import torch
import random
import heapq
import math
from tqdm import tqdm
import numpy as np
import argparse

from train import INPUT_DIM, OUTPUT_DIM,NUM_LAYERS,L,PATH,M,HIDDEN_SIZE,NUM_GNN_LAYERS,DROP_OUT
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

# 标准化
"""
adj_mean = (M - 1) / 2
adj_std = math.sqrt(((M - 1) ** 2) / 12)
coord_mean = (L - 1) / 2
coord_std = math.sqrt((L ** 2 - 1) / 12)
mean = torch.tensor(
    [adj_mean] * (L * L) + [coord_mean] * 2,
    dtype=torch.float32
)
std = torch.tensor(
    [adj_std] * (L * L) + [coord_std] * 2,
    dtype=torch.float32
)
"""

# 模型参数
model_args={
    "L":L,
    "in_dim":INPUT_DIM,
    "out_dim":OUTPUT_DIM,
    "num_layers":NUM_LAYERS,
    "hidden_size":HIDDEN_SIZE,
    "num_gnn_layers":NUM_GNN_LAYERS,
    "dropout":DROP_OUT,
}

def generate_random_adjacency(n, rng=None,low=1, high=10):
    """生成 n×n 随机邻接矩阵（完全图，正权）"""
    if low == 0:
        low = 1e-18  # avoid zero weights
    mat = rng.uniform(low, high, size=(n, n)).astype(np.float32)
    np.fill_diagonal(mat, 0)
    return mat

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
    model = MyClassifier(**model_args).to(DEVICE)
    model.load_state_dict(torch.load(PATH, map_location=DEVICE))
    model.eval()
    print("模型权重加载成功")

    # ---------- 推理 ----------
    TEST_LEN = int(1e4)

    model.eval()
    right=0
    origin_right=0

    with torch.no_grad():
        for _ in tqdm(range(TEST_LEN),mininterval=1):
            while(True):
                #adj = generate_random_adjacency(L,rng=IT_rng,low=1,high=M)
                adj = generate_random_adjacency(L, rng=IT_rng,low=0,high=M-1)
                #start,end = IT_rng.sample(range(L), 2)
                start, end = IT_rng.choice(L, size=2, replace=False)
                Min,path = dijkstra(adj,start,end)
                assert len(path)>1, "fix me"
                if len(path)==2:
                    continue #只要路径长度大于等于3的
                else:
                    break
            #flat = [val for row in adj for val in row] + [start, end]
            flat = adj.flatten().tolist() + [start, end]
            map_tensor = torch.tensor(flat, dtype=torch.float32)
            #map_tensor = (map_tensor - mean) / (std + 1e-8)
            batch=map_tensor.unsqueeze(0)
            ans = path[1]
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
