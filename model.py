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
                in_channels=1):
        super().__init__()
        self.num_poolings = num_poolings
        self.L = L
        self.in_channels=in_channels
        self.init_channels=init_channels
        self.final_conv_channels=final_conv_channels
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
        conv_feat = self.conv(grid)
        conv_feat = conv_feat.view(conv_feat.size(0), -1)
        out=conv_feat
        return out

class MyClassifier(nn.Module):
    def __init__(self, *,in_dim, out_dim, num_layers, hidden_size, num_poolings, dropout,L,
                extra_dim=2,
                # -------new-added CNN's args--------
                init_channels=32,          # 初始卷积核数
                final_conv_channels=128,   # 最后卷积输出通道数
                in_channels=1):
        super().__init__()
        self.extra_dim=extra_dim
        self.L = L
        assert self.L**2+self.extra_dim==in_dim , "Error in parameter calculation on the training end"
        self.Cnn=CnnBlock(num_poolings=num_poolings, 
                    L=L,
                    final_conv_channels=final_conv_channels,
                    in_channels=in_channels,
                    init_channels=init_channels)
        # --------------MLP---------------
        prev_dim = self.Cnn.conv_out_dim + in_dim
        self.coeff = nn.Parameter(torch.tensor(1.0))
        # MLP的输入既包括CNN的输出也包括初始数据
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
        Cnn_out=self.Cnn(x)
        combined = torch.cat([Cnn_out, self.coeff*x], dim=1) 
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