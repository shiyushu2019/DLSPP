import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
class AttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, *, use_residual=True, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.q_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, in_dim, bias=False)
        self.linear = nn.Linear(in_dim, out_dim)   # 对聚合后的特征做变换
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.use_residual = use_residual and (in_dim == out_dim)
        self.scale = in_dim ** -0.5

    def forward(self, x, mask=None):
        Q = self.q_proj(x)  
        K = self.k_proj(x) 
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1) 
        agg = torch.bmm(attn, x) 
        out = self.norm(agg)
        out = self.linear(out)
        out = self.act(out)
        out = self.drop(out)
        if self.use_residual:
            out = out + x
        return out

class TransformerBlock(nn.Module):
    def __init__(self  ,*,out_dim, L,hidden_size, use_residual=True, dropout, num_transformer_layers):
        super().__init__()
        assert num_transformer_layers>0
        self.num_transformer_layers=num_transformer_layers
        self.register_buffer('eye', torch.eye(L))
        self.L=L
        self.out_dim=out_dim
        transformer_in = 4* self.L
        transformer_out = hidden_size
        self.transformer_layers = nn.ModuleList()
        for i in range(self.num_transformer_layers):
            layer_in = transformer_in if i == 0 else transformer_out
            self.transformer_layers.append(AttentionLayer(layer_in, transformer_out, use_residual=True, dropout=dropout))
        self.transformer_to_mlp = nn.Sequential(
            nn.Linear(2 * transformer_out, out_dim),
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
        start_onehot = F.one_hot(start_idx, num_classes=L).float().unsqueeze(1)  
        end_onehot   = F.one_hot(end_idx,   num_classes=L).float().unsqueeze(1) 
        start_expand = start_onehot.expand(-1, L, -1)   
        end_expand   = end_onehot.expand(-1, L, -1)    
        # 拼接进节点特征
        h = torch.cat([self.eye.unsqueeze(0).expand(batch_size, -1, -1), adj, start_expand, end_expand], dim=-1)
        eps = 1e-12     
        for layer in self.transformer_layers:
            h = layer(h) 
        idx = torch.arange(batch_size, device=x.device)
        start_feat = h[idx, start_idx, :]  
        end_feat   = h[idx, end_idx, :]   
        combined = torch.cat([start_feat, end_feat], dim=-1)        
        out = self.transformer_to_mlp(combined)                 
        return out

class MyClassifier(nn.Module):
    def __init__(self, *,in_dim, out_dim, num_layers, hidden_size,  dropout,L,
                extra_dim=2,
                use_cnn,
                use_transformer,
                use_direct,
                # --------CNN---------
                init_channels=32,          # 初始卷积核数
                final_conv_channels=128,   # 最后卷积输出通道数
                in_channels=1,
                num_poolings=1,
                do_norm,
                # -------transformer--------
                num_transformer_layers=1,
                transformer_hidden_size=512,
                transformer_out_dim=None
                ):
        # 所有没有默认值的命名参数已确认正确传递
        super().__init__()
        if transformer_out_dim==None:
            transformer_out_dim=L*L
        self.extra_dim=extra_dim
        self.L = L
        self.use_transformer=use_transformer
        self.use_cnn=use_cnn
        self.use_direct=use_direct
        assert use_cnn or use_transformer or use_direct , "At least one channl is used"
        assert self.L**2+self.extra_dim==in_dim , "Error in parameter calculation on the training end"
        if use_cnn:
            self.Cnn=CnnBlock(num_poolings=num_poolings, 
                    L=L,
                    final_conv_channels=final_conv_channels,
                    in_channels=in_channels,
                    init_channels=init_channels,
                    do_norm=do_norm)
        if use_transformer:
            self.transformer=TransformerBlock(L=self.L,hidden_size=transformer_hidden_size,dropout=dropout,out_dim=transformer_out_dim,
                        num_transformer_layers=num_transformer_layers)
        
        # --------------MLP---------------
        prev_dim = 0
        if use_direct:
            prev_dim += in_dim
        if use_cnn:
            prev_dim += self.Cnn.conv_out_dim
        if use_transformer:
            prev_dim += self.transformer.out_dim
        if not use_direct:
            prev_dim += extra_dim # 没有MLP直连通道，就要额外加上起点和终点
        self.coeff_cnn = nn.Parameter(torch.tensor(1.0))
        self.coeff_transformer = nn.Parameter(torch.tensor(1.0))
        self.coeff_direct = nn.Parameter(torch.tensor(1.0))
        # MLP的输入既包括CNN、transformer的输出也包括初始数据
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
        outputs = []
        if self.use_direct:
            outputs.append(x*self.coeff_direct)
        if self.use_cnn:
            outputs.append(self.coeff_cnn * self.Cnn(x))
        if self.use_transformer:
            outputs.append(self.coeff_transformer * self.transformer(x))
        if not self.use_direct:
            outputs.append(x[:, self.L*self.L:]) # 没有MLP直连通道，就要额外加上起点和终点
        combined = torch.cat(outputs, dim=1) 
        out = self.mlp(combined)
        return out

if __name__ == "__main__":
    from train import model_args, cnn_config, transformer_config
    model = MyClassifier(**model_args, **cnn_config, **transformer_config).to("cpu")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型参数量级: {total_params / 1e9:.2f} B")  