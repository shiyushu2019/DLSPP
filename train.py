#export CUDA_VISIBLE_DEVICES=2

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from DataGenerator import FakeList, MapRouteDataset

PATH="checkpoint/model.pth"
RESUME_FROM=None # pretrain weight or None
LEN=int(5e8)
L = 10
M = 10
INPUT_DIM = L * L + 2 
OUTPUT_DIM = L     
EPOCHS = 1000
PATIENCE=3000000      
MOMENTUM = 0.9 

LR = 1e-3
DROP_OUT=0.0
WEIGHT_DECAY = 0.0
BATCH_SIZE = 512
          
NUM_LAYERS = 10
HIDDEN_SIZE=int(4096*3)
NUM_POOLINGS=1

NUM_WORKERS=48
PREFETCH_FACTOR=2

VAL_STEP=int(3000)
MININTERVAL=10
REVERSE_G=25

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------- 模型定义 ----------
import torch
import torch.nn as nn

class MyClassifier(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers):
        super().__init__()
        self.num_poolings = NUM_POOLINGS

        conv_layers = []
        in_ch = 1
        out_ch = 32
        for i in range(NUM_POOLINGS):
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
        in_ch = 128   #

        self.conv = nn.Sequential(*conv_layers)

        out_size = L // (2 ** NUM_POOLINGS)
        assert out_size > 0, f"池化次数 {NUM_POOLINGS} 过大"
        self.conv_out_dim = in_ch * out_size * out_size
        extra_dim = in_dim - L * L                        
        prev_dim = self.conv_out_dim + extra_dim  

        # 后续 MLP

        hidden_size = HIDDEN_SIZE   # 可调整，也可作为参数传入
        layers = []
        # 如果 num_layers == 1，则直接输出（但通常不会，因为任务复杂）
        for i in range(num_layers - 1):
            layers.append(nn.Linear(prev_dim, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(p=DROP_OUT))
            prev_dim = hidden_size
        layers.append(nn.Linear(prev_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        grid = x[:, :L*L].view(-1, 1, L, L)   # (batch, 1, 50, 50)
        extra = x[:, L*L:]                       # (batch, 2)
        conv_feat = self.conv(grid)               # (batch, 128, 12, 12)
        conv_feat = conv_feat.view(conv_feat.size(0), -1)  # flatten
        combined = torch.cat([conv_feat, extra], dim=1)    # (batch, conv_out_dim+2)
        out = self.mlp(combined)
        return out

if __name__ == "__main__":
    reserved_tensor = torch.empty(1024 * 1024 *1024*REVERSE_G, dtype=torch.uint8).cuda() #25G
    
    seed_bia=1
    fakelist=FakeList(M,L,LEN,seed_bia)
    dataset = MapRouteDataset(M,L,fakelist)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,prefetch_factor=PREFETCH_FACTOR)
    seed_bia=2
    val_fakelist=FakeList(M,L,10000,seed_bia)
    val_dataset = MapRouteDataset(M,L,val_fakelist)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,prefetch_factor=PREFETCH_FACTOR)

    print(f"loader generated")

    val_len = len(val_dataset)
    train_len = len(dataset)

    print(f"训练集大小: {train_len}, 验证集大小: {val_len}")
    print("loader done")

    #  初始化模型、损失函数、优化器
    model = MyClassifier(INPUT_DIM, OUTPUT_DIM, NUM_LAYERS).to(DEVICE)
    if RESUME_FROM:
        model.load_state_dict(torch.load(RESUME_FROM))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY )
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # 训练循环
    patience = PATIENCE
    trigger_times = 0  # 连续不提升次数
    best_val_acc = 0.0

    flag=False
    for epoch in range(EPOCHS):
        if flag :
            break
        model.train()
        loop = tqdm(loader, desc=f'Epoch {epoch+1}/{EPOCHS}',mininterval=MININTERVAL)

        batch_cnt=0
        cur_loss = 0.0
        
        for batch_x, batch_y in loop:
            model.train()
            
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            logits = model(batch_x)
            loss = criterion(logits, batch_y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cur_loss += loss.item()
            batch_cnt+=1
            avg_loss = cur_loss / batch_cnt

            if batch_cnt == VAL_STEP:
                batch_cnt=0
                cur_loss=0.0
                #--------eval---------
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                        logits = model(batch_x)
                        loss = criterion(logits, batch_y.long())
                        val_loss += loss.item()
                        
                        # 计算准确率
                        preds = torch.argmax(logits, dim=1)
                        correct += (preds == batch_y).sum().item()
                        total += batch_y.size(0)
                
                avg_val_loss = val_loss / len(val_loader)
                val_acc = correct / total

                print(f"Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

                #scheduler.step(val_acc) 
                # ---------- 保存最佳模型 ----------
                if val_acc > best_val_acc:
                    trigger_times = 0   # 重置
                    best_val_acc = val_acc
                    torch.save(model.state_dict(),  PATH)
                    print(f" 验证准确率提升至 {val_acc:.4f}, 已保存")
                else:
                    trigger_times += 1
                    if trigger_times >= patience:
                        print(f"验证准确率连续 {patience} 个 epoch 未提升，提前终止")
                        flag=True
                        break
        
    

        