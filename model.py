
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        # 标准 Pre-Norm 结构：先归一化，再线性变换，再激活
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        # 1. 先对输入 x 做 LayerNorm（关键改动）
        out = self.norm(x)
        # 2. 再做线性变换
        out = self.linear(out)
        # 3. 激活
        out = self.act(out)
        # 4. Dropout（如果有）
        out = self.drop(out)
        # 5. 残差连接
        return residual + out
        
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, *, use_residual=True, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)         
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_residual = use_residual and (in_dim == out_dim)
        
    def forward(self, x, norm_A):
        agg = torch.bmm(norm_A, x)           
        out = self.norm(agg)                     
        out = self.linear(out)                   
        out = self.act(out)
        out = self.drop(out)
        if self.use_residual:
            out = out + x
        return out

class MyClassifier(nn.Module):
    def __init__(self, *,in_dim, out_dim, num_layers, hidden_size, num_gnn_layers, dropout,L):
        super().__init__()
        self.num_gnn_layers = num_gnn_layers
        self.L = L
        assert (self.L * self.L + 2) == in_dim, "in_dim must be L*L + 2"
        assert num_layers >= 2, "num_layers must be at least 2"

        # ------ GNN ------
        if self.num_gnn_layers > 0:
            gnn_in = self.L
            gnn_out = hidden_size
            self.gnn_layers = nn.ModuleList()
            for i in range(self.num_gnn_layers):
                layer_in = gnn_in if i == 0 else gnn_out
                self.gnn_layers.append(GCNLayer(layer_in, gnn_out, use_residual=True, dropout=dropout))
            self.gnn_to_mlp = nn.Sequential(
                nn.Linear(2 * gnn_out, in_dim),
                nn.SiLU(),                         
                nn.Dropout(dropout),              
                nn.LayerNorm(in_dim)              
            )
        else:
            self.gnn_layers = None
            self.gnn_to_mlp = None
        self.register_buffer('eye', torch.eye(L))   # shape (L, L)

        # ------ MLP ---------
        prev_dim = in_dim
        mlp_modules = []
        mlp_modules.append(nn.LayerNorm(prev_dim))         
        mlp_modules.append(nn.Linear(prev_dim, hidden_size))
        mlp_modules.append(nn.SiLU())
        mlp_modules.append(nn.Dropout(p=dropout))          
        for _ in range(num_layers - 2):
            mlp_modules.append(ResidualBlock(hidden_size, dropout))
        mlp_modules.append(nn.LayerNorm(hidden_size))
        mlp_modules.append(nn.Linear(hidden_size, out_dim))
        self.mlp = nn.Sequential(*mlp_modules)

    def forward(self, x):
        if self.num_gnn_layers > 0:
            batch_size = x.size(0)
            L = self.L 
            adj_flat = x[:, :L*L]
            adj = adj_flat.view(batch_size, L, L)
            start_idx = x[:, L*L].long()
            end_idx   = x[:, L*L+1].long()
            start_idx = torch.clamp(start_idx, 0, L-1)
            end_idx   = torch.clamp(end_idx, 0, L-1)
            h = adj 
            A_hat = adj + self.eye.unsqueeze(0)    
            D_hat = A_hat.sum(dim=-1, keepdim=True) 
            eps = 1e-12                              # [MODIFIED] 增加极小值防止除零
            #D_inv_sqrt = (D_hat + eps) ** -0.5       # (B, L, 1)
            #norm_A_sym = D_inv_sqrt * A_hat * D_inv_sqrt.transpose(-1, -2)  # (B, L, L)  
            D_inv = 1.0 / (D_hat + eps)                      # (B, L, 1)
            norm_A = D_inv * A_hat                 
            for layer in self.gnn_layers:
                h = layer(h, norm_A)  
            idx = torch.arange(batch_size, device=x.device)
            start_feat = h[idx, start_idx, :]  
            end_feat   = h[idx, end_idx, :]   
            combined = torch.cat([start_feat, end_feat], dim=-1)        
            x_gnn = self.gnn_to_mlp(combined)    
            """           
            x_gnn[:, :L*L] = x_gnn[:, :L*L] + x[:, :L*L]   
            x_gnn[:, L*L:] = x[:, L*L:]    
            """  
            x_gnn = x_gnn + x              
            out = self.mlp(x_gnn)
        else:
            out = self.mlp(x)
        return out

if __name__ == "__main__":
    from train import model_args
    model = MyClassifier(**model_args).to("cpu")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型参数量级: {total_params / 1e9:.2f} B")  