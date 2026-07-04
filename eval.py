import torch
from train import MyClassifier, INPUT_DIM, OUTPUT_DIM, DEVICE,NUM_LAYERS,LEN,BATCH_SIZE,L,PATH,M
import random
import heapq
import math
from typing import Union, List, Sequence
from tqdm import tqdm


IT_rng = random.Random(int(1e19))
# 1. 邻接矩阵统计量 (依赖 M)
adj_mean = (1 + M) / 2
adj_std = math.sqrt((M**2 - 1) / 12)

# 2. 坐标统计量 (依赖 L)
coord_mean = (L - 1) / 2
coord_std = math.sqrt((L**2 - 1) / 12)

# 3. 拼接成完整张量 (自动适应 L 的变化)
mean = torch.tensor(
    [adj_mean] * (L * L) + [coord_mean] * 2,
    dtype=torch.float32
)
std = torch.tensor(
    [adj_std] * (L * L) + [coord_std] * 2,
    dtype=torch.float32
)

def generate_random_adjacency(n, rng=None,seed_bia=1,low=1, high=10):
    """生成 n×n 随机邻接矩阵（完全图，正权）"""
    mat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                #for _ in range(seed_bia):
                #    mat[i][j] = rng.randint(low, high)
                mat[i][j] = rng.randint(low, high)
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
    model = MyClassifier(INPUT_DIM, OUTPUT_DIM, NUM_LAYERS).to(DEVICE)
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
                adj = generate_random_adjacency(L,rng=IT_rng,seed_bia=3,low=1,high=M)
                start,end = IT_rng.sample(range(L), 2)
                Min,path = dijkstra(adj,start,end)
                assert len(path)>1, "fix me"
                if len(path)==2:
                    continue #只要路径长度大于等于3的
                else:
                    break
            flat = [val for row in adj for val in row] + [start, end]
            map_tensor = torch.tensor(flat, dtype=torch.float32)
            map_tensor = (map_tensor - mean) / (std + 1e-8)
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
                assert ans_length >= Min , "fix me"
                if ans_length==Min:
                    right+=1


    acc =right / TEST_LEN
    origin_acc=origin_right / TEST_LEN
    print(f"acc: {acc:.4f},origin_acc: {origin_acc:.4f}")
