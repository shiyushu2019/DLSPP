import random
import heapq
import torch
import math

def generate_random_adjacency(n, rng=None,seed_bia=1,low=1, high=10):
    """生成 n×n 随机邻接矩阵（完全图，正权）"""
    mat = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                #for _ in range(seed_bia):
                #    mat[i][j] = rng.randint(low, high)
                x=rng.uniform(low, high)
                while x==0:
                    x=rng.uniform(low, high)
                mat[i][j] = x
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

class FakeList():
    def __init__(self,M,L,length,seed_bia) -> None:
        self.length=length
        self.seed_bia=seed_bia
        self.L=L
        self.M=M
    def __len__(self):
        return self.length
    def __getitem__(self,index):
        if index>self.length-1:
            raise ValueError("Index out of range")

        rng = random.Random(index+int(1e18)*self.seed_bia)   # 独立随机生成器

        n = self.L # 节点数

        while True:
            #start, end = random.sample(range(n), 2)
            #adj = generate_random_adjacency(n)
            #for _ in range(self.seed_bia):
            #    start, end = rng.sample(range(n), 2)
            start, end = rng.sample(range(n), 2)
            adj = generate_random_adjacency(n, rng=rng,seed_bia=self.seed_bia,low=0,high=self.M-1)
            _, path = dijkstra(adj, start, end)  # 完全图保证路径存在

            assert len(path)>1, "fix me"
            if len(path)==2:
                continue #只要路径长度大于等于3的
            
            # 按照格式：map = [邻接矩阵, [起点, 终点]]
            return {
                "map": [adj, [start, end]],
                "route": path
            }

class MapRouteDataset:
    def __init__(self, M,L,fakelist):
        self.fakelist = fakelist
        self.L=L
        self.M = M

        adj_mean = (M - 1) / 2
        adj_std = math.sqrt(((M - 1) ** 2) / 12)
        coord_mean = (L - 1) / 2
        coord_std = math.sqrt((L ** 2 - 1) / 12)
        self.mean = torch.tensor(
            [adj_mean] * (L * L) + [coord_mean] * 2,
            dtype=torch.float32
        )
        self.std = torch.tensor(
            [adj_std] * (L * L) + [coord_std] * 2,
            dtype=torch.float32
        )
        
    def __len__(self):
        return len(self.fakelist)
        

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError("Index out of range")

        sample = self.fakelist[index]          # 只生成一次
        
        # 展平 map：邻接矩阵 (10x10) + [start, end]
        adj, [start, end] = sample["map"]
        flat = [val for row in adj for val in row] + [start, end]
        map_tensor = torch.tensor(flat, dtype=torch.float32)
        map_tensor = (map_tensor - self.mean) / (self.std + 1e-8)
        
        # route 的第二个元素（第1个中间节点）
        route_tensor = torch.tensor(sample["route"][1], dtype=torch.float32)
        
        return map_tensor, route_tensor
