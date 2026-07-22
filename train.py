#export CUDA_VISIBLE_DEVICES=2

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import argparse

from DataGenerator import FakeList, MapRouteDataset
from model import MyClassifier

#----------调试------------
DEBUG = False
USE_CNN = True
USE_TRANSFORMER = True
USE_GNN = True
USE_DIRECT = True
PATH="checkpoint/CG/model.pth"
RESUME_FROM="checkpoint/CG/model.pth" # pretrain weight or None
LEN=int(1e9)

#---------数据参数------------
DO_STD=False
MIN_JUMP=1

#---------训练参数------------
LR = 1e-3
BATCH_SIZE = 512
DROP_OUT=0.0
WEIGHT_DECAY = 0.0
EPOCHS = 1000
PATIENCE=3000000      
MOMENTUM = 0.9
SCH_PATIENCE=8   # 验证准确率连续不下降就降低LR
SCH_FACTOR=0.5   # 降低的比例

#---------模型参数------------
L = 10
M = 10
INPUT_DIM = L * L + 2 
OUTPUT_DIM = L     
NUM_LAYERS = 10
HIDDEN_SIZE=int(4096*2)

#---------硬参数------------
NUM_WORKERS=48
PREFETCH_FACTOR=2
VAL_STEP=int(3000)
MININTERVAL=60
REVERSE_G=0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_args={
    "L":L,
    "in_dim":INPUT_DIM,
    "out_dim":OUTPUT_DIM,
    "num_layers":NUM_LAYERS,
    "hidden_size":HIDDEN_SIZE,
    "dropout":DROP_OUT,
    "extra_dim":2,
    "use_cnn":USE_CNN,
    "use_transformer":USE_TRANSFORMER,
    "use_direct":USE_DIRECT,
    "use_gnn":USE_GNN
}
cnn_config={
    "init_channels":32,         
    "final_conv_channels":128,
    "in_channels":1,
    "num_poolings":1,
    "do_norm":True
}
transformer_config={
    "num_transformer_layers":3,
    "transformer_hidden_size":512,
    "transformer_out_dim":2048
}
gnn_config={
    "num_gnn_layers":3,
    "gnn_hidden_size":512,
    "gnn_out_dim":2048
}

if __name__ == "__main__":
    # 抢占富裕显存避免降速
    reserved_tensor = torch.empty(1024 * 1024 *1024*REVERSE_G, dtype=torch.uint8).cuda()
    if DEBUG:
        MININTERVAL=1
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--begin', type=int, default=0) # 从现有权重开始训练时，从第x条数据开始
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--min_jump', type=int, default=MIN_JUMP)
    args = parser.parse_args()
    LR=args.lr
    MIN_JUMP=args.min_jump
    
    seed_bia=1
    fakelist=FakeList(M,L,LEN,seed_bia,min_jump=MIN_JUMP)
    dataset = MapRouteDataset(M,L,fakelist,do_std=DO_STD)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,prefetch_factor=PREFETCH_FACTOR)
    seed_bia=2
    val_fakelist=FakeList(M,L,10000,seed_bia,min_jump=MIN_JUMP)
    val_dataset = MapRouteDataset(M,L,val_fakelist,do_std=DO_STD)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,prefetch_factor=PREFETCH_FACTOR)

    val_len = len(val_dataset)
    train_len = len(dataset)

    print(f"训练集大小: {train_len}, 验证集大小: {val_len}")

    #  初始化模型、损失函数、优化器
    model = MyClassifier(**model_args, **cnn_config, **transformer_config,**gnn_config).to(DEVICE)
    if RESUME_FROM:
        model.load_state_dict(torch.load(RESUME_FROM))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY )

    # -----------自动降LR---------------
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=SCH_FACTOR, patience=SCH_PATIENCE)

    # 训练循环
    patience = PATIENCE
    trigger_times = 0  # 连续不提升次数
    best_val_acc = 0.0

    if RESUME_FROM:
        begin=args.begin
        print(f"从第 {begin} 条数据接续训练")
        from torch.utils.data import Subset
        subset = Subset(dataset, range(begin, train_len))  # 不加载数据
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,prefetch_factor=PREFETCH_FACTOR)  # 依然懒加载
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
        best_val_acc=val_acc
        scheduler.step(val_acc) 
        print(f"resumed from checkpint. best_val_acc: {best_val_acc}")

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

            if DEBUG:
                loop.set_postfix(loss=avg_loss)

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


                # -----------自动降LR---------------
                old_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_acc) 
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    print(f"学习率已调整：{old_lr:.6f} -> {new_lr:.6f}")


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