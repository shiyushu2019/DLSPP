
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

class MyClassifier(nn.Module):
    def __init__(self, *,in_dim, out_dim, num_layers, hidden_size, num_poolings, dropout,L):
        super().__init__()
        self.num_poolings = num_poolings
        self.L = L
        conv_layers = []
        in_ch = 1
        out_ch = 32
        for i in range(num_poolings):
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            conv_layers.append(nn.SiLU())
            in_ch = out_ch
            out_ch = out_ch * 2
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            conv_layers.append(nn.SiLU())
            in_ch = out_ch
            conv_layers.append(nn.MaxPool2d(2))

        conv_layers.append(nn.Conv2d(in_ch, 128, kernel_size=3, padding=1))
        conv_layers.append(nn.SiLU())
        in_ch = 128   

        self.conv = nn.Sequential(*conv_layers)

        out_size = L // (2 ** num_poolings)
        assert out_size > 0, f"池化次数 {num_poolings} 过大"
        self.conv_out_dim = in_ch * out_size * out_size
        extra_dim = in_dim - L * L                        
        prev_dim = self.conv_out_dim + extra_dim  

        # 后续 MLP
        hidden_size = hidden_size
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
        grid = x[:, :self.L*self.L].view(-1, 1, self.L, self.L)   # (batch, 1, 50, 50)
        extra = x[:, self.L*self.L:]                       # (batch, 2)
        conv_feat = self.conv(grid)               # (batch, 128, 12, 12)
        conv_feat = conv_feat.view(conv_feat.size(0), -1)  # flatten
        combined = torch.cat([conv_feat, extra], dim=1)    # (batch, conv_out_dim+2)
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