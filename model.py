
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 2)
        self.linear2 = nn.Linear(dim * 2, dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        out = self.norm(x)
        out = self.linear1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.linear2(out)
        out = self.drop(out) 
        return residual + out

class MyClassifier(nn.Module):
    def __init__(self, *,in_dim, out_dim, num_layers, hidden_size, dropout):
        super().__init__()
        prev_dim = in_dim
        mlp_modules = []
        mlp_modules.append(nn.LayerNorm(prev_dim))         
        mlp_modules.append(nn.Linear(prev_dim, hidden_size))
        mlp_modules.append(nn.SiLU())
        mlp_modules.append(nn.Dropout(p=dropout))          
        assert num_layers%2==0, "ResidualBlock has two linear, so num_layers should be even number."
        num_residual_num=int((num_layers-2)/2) 
        for _ in range(num_residual_num):
            mlp_modules.append(ResidualBlock(hidden_size, dropout))
        mlp_modules.append(nn.LayerNorm(hidden_size))
        mlp_modules.append(nn.Linear(hidden_size, out_dim))
        self.mlp = nn.Sequential(*mlp_modules)

    def forward(self, x):
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