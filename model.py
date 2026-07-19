import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        residual = x
        out = self.norm(x)
        out = self.linear(out)
        out = self.act(out)
        out = self.drop(out)
        return residual + out

class CnnBlock(nn.Module):
    def __init__(self, * ,num_poolings,L,
                init_channels=32,          # 初始卷积核数
                final_conv_channels=128,   # 最后卷积输出通道数
                in_channels=1,
                do_norm=True):
        super().__init__()
        self.num_poolings = num_poolings
        self.L = L
        self.in_channels=in_channels
        self.init_channels=init_channels
        self.final_conv_channels=final_conv_channels
        self.input_norm = nn.InstanceNorm2d(in_channels, affine=True)
        self.do_norm=do_norm
        conv_layers = []
        in_ch = in_channels
        out_ch = init_channels
        for i in range(num_poolings):
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            conv_layers.append(nn.SiLU())
            in_ch = out_ch
            out_ch = out_ch * 2
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            conv_layers.append(nn.SiLU())
            in_ch = out_ch
            conv_layers.append(nn.MaxPool2d(2))
        conv_layers.append(nn.Conv2d(in_ch, final_conv_channels, kernel_size=3, padding=1))
        conv_layers.append(nn.SiLU())
        in_ch = final_conv_channels 
        self.conv = nn.Sequential(*conv_layers)
        out_size = L // (2 ** num_poolings)
        assert out_size > 0, f"池化次数 {num_poolings} 过大"
        self.conv_out_dim = in_ch * out_size * out_size                     
    def forward(self,x):
        grid = x[:, :self.L*self.L].view(-1, self.in_channels, self.L, self.L)
        if self.do_norm:
            grid = self.input_norm(grid)
        conv_feat = self.conv(grid)
        conv_feat = conv_feat.view(conv_feat.size(0), -1)
        out=conv_feat
        return out
        
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

class GnnBlock(nn.Module):
    def __init__(self  ,*,out_dim, L,hidden_size, use_residual=True, dropout, num_gnn_layers):
        super().__init__()
        assert num_gnn_layers>0
        self.num_gnn_layers=num_gnn_layers
        self.register_buffer('eye', torch.eye(L))
        self.L=L
        self.out_dim=out_dim
        gnn_in = self.L
        gnn_out = hidden_size
        self.gnn_layers = nn.ModuleList()
        for i in range(self.num_gnn_layers):
            layer_in = gnn_in if i == 0 else gnn_out
            self.gnn_layers.append(GCNLayer(layer_in, gnn_out, use_residual=True, dropout=dropout))
        self.gnn_to_mlp = nn.Sequential(
            nn.Linear(2 * gnn_out, out_dim),
            nn.SiLU(),                         
            nn.Dropout(dropout),              
            nn.LayerNorm(out_dim)
        )
    def forward(self, x):
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
        eps = 1e-12
        D_inv = 1.0 / (D_hat + eps)      
        norm_A = D_inv * A_hat                 
        for layer in self.gnn_layers:
            h = layer(h, norm_A)  
        idx = torch.arange(batch_size, device=x.device)
        start_feat = h[idx, start_idx, :]  
        end_feat   = h[idx, end_idx, :]   
        combined = torch.cat([start_feat, end_feat], dim=-1)        
        out = self.gnn_to_mlp(combined)                 
        return out

class MyClassifier(nn.Module):
    def __init__(self, *,in_dim, out_dim, num_layers, hidden_size,  dropout,L,
                extra_dim=2,
                use_cnn=True,
                use_gnn=True,
                # --------CNN---------
                init_channels=32,          # 初始卷积核数
                final_conv_channels=128,   # 最后卷积输出通道数
                in_channels=1,
                num_poolings=1,
                # -------GNN--------
                num_gnn_layers=1,
                gnn_hidden_size=512,
                gnn_out_dim=None
                ):
        super().__init__()
        if gnn_out_dim==None:
            gnn_out_dim=L*L
        self.extra_dim=extra_dim
        self.L = L
        self.use_gnn=use_gnn
        self.use_cnn=use_cnn
        assert self.L**2+self.extra_dim==in_dim , "Error in parameter calculation on the training end"
        if use_cnn:
            self.Cnn=CnnBlock(num_poolings=num_poolings, 
                    L=L,
                    final_conv_channels=final_conv_channels,
                    in_channels=in_channels,
                    init_channels=init_channels)
        if use_gnn:
            self.Gnn=GnnBlock(L=self.L,hidden_size=gnn_hidden_size,dropout=dropout,out_dim=gnn_out_dim,
                        num_gnn_layers=num_gnn_layers)
        
        # --------------MLP---------------
        prev_dim = in_dim
        if use_cnn:
            prev_dim += self.Cnn.conv_out_dim
        if use_gnn:
            prev_dim += self.Gnn.out_dim
        self.coeff_cnn = nn.Parameter(torch.tensor(1.0))
        self.coeff_gnn = nn.Parameter(torch.tensor(1.0))
        # MLP的输入既包括CNN、GNN的输出也包括初始数据
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
        outputs = [x]
        if self.use_cnn:
            outputs.append(self.coeff_cnn * self.Cnn(x))
        if self.use_gnn:
            outputs.append(self.coeff_gnn * self.Gnn(x))
        combined = torch.cat(outputs, dim=1) 
        out = self.mlp(combined)
        return out

if __name__ == "__main__":
    from train import model_args
    model = MyClassifier(**model_args).to("cpu")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型参数量级: {total_params / 1e9:.2f} B")  