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
PATH="checkpoint/CG-uniform/model.pth"
RESUME_FROM= "checkpoint/CG-uniform/model.pth" # pretrain weight or None
LEN=int(1e9)

#---------数据参数------------
DO_STD=False

#---------训练参数------------
LR = 5e-5
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
HIDDEN_SIZE=int(4096*3)

#---------硬参数------------
NUM_WORKERS=12
PREFETCH_FACTOR=2
VAL_STEP=int(3000)
MININTERVAL=60
REVERSE_G=20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--begin', type=int, default=0) # 从现有权重开始训练时，从第x条数据开始
parser.add_argument('--lr', type=float, default=LR)
parser.add_argument('--debug', type=bool, default=DEBUG)
parser.add_argument('--reserve', type=int, default=0)

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
    args = parser.parse_args()
    LR=args.lr
    DEBUG=args.debug
    REVERSE_G=args.reserve
    
    # 抢占富裕显存避免降速
    reserved_tensor = torch.empty(1024 * 1024 *1024*REVERSE_G, dtype=torch.uint8).cuda()
    if DEBUG:
        MININTERVAL=1
    
    #  我们要让high_jump在 1，2，3，4中均匀选取,每个占四分之一
    assert BATCH_SIZE%4==0
    assert LEN%4==0

    train_loaders=list()
    for jump,seed_bia in zip([1,2,3,4],[1,2,3,4]):
        fakelist=FakeList(M,L,LEN//4,seed_bia,min_jump=jump)
        dataset = MapRouteDataset(M,L,fakelist,do_std=DO_STD)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE//4, shuffle=False, num_workers=NUM_WORKERS,prefetch_factor=PREFETCH_FACTOR)
        train_loaders.append(loader)
    
    val_loaders=list()
    val_len=10000
    assert val_len%4==0
    for jump,seed_bia in zip([1,2,3,4],[5,6,7,8]):
        fakelist=FakeList(M,L,val_len//4,seed_bia,min_jump=jump)
        dataset = MapRouteDataset(M,L,fakelist,do_std=DO_STD)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE//4, shuffle=False, num_workers=NUM_WORKERS,prefetch_factor=PREFETCH_FACTOR)
        val_loaders.append(loader)

    print(f"训练集大小: {LEN}, 验证集大小: {val_len}")

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
        pair_num=begin//4  # 忽略除4误差
        train_loaders=list()
        for jump,seed_bia in zip([1,2,3,4],[1,2,3,4]):
            fakelist=FakeList(M,L,LEN//4,seed_bia,min_jump=jump)
            dataset = MapRouteDataset(M,L,fakelist,do_std=DO_STD)
            dataset = Subset(dataset, range(pair_num, LEN//4))  # 取子集
            loader = DataLoader(dataset, batch_size=BATCH_SIZE//4, shuffle=False, num_workers=NUM_WORKERS,prefetch_factor=PREFETCH_FACTOR)
            train_loaders.append(loader)

        #--------eval---------
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for (batch1_x, batch1_y),(batch2_x, batch2_y),(batch3_x, batch3_y),(batch4_x, batch4_y) in zip(*val_loaders):
                batch_x = torch.cat([batch1_x, batch2_x, batch3_x, batch4_x], dim=0)
                batch_y = torch.cat([batch1_y, batch2_y, batch3_y, batch4_y], dim=0)
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                logits = model(batch_x)
                loss = criterion(logits, batch_y.long())
                val_loss += loss.item()
                
                # 计算准确率
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        
        avg_val_loss = val_loss / len(val_loaders[0])
        val_acc = correct / total
        best_val_acc=val_acc
        scheduler.step(val_acc) 
        print(f"resumed from checkpint. best_val_acc: {best_val_acc}")

    flag=False
    for epoch in range(EPOCHS):
        if flag :
            break
        model.train()

        loop = tqdm(zip(*train_loaders), desc=f'Epoch {epoch+1}/{EPOCHS}',mininterval=MININTERVAL)

        batch_cnt=0
        cur_loss = 0.0
        
        for (batch1_x, batch1_y),(batch2_x, batch2_y),(batch3_x, batch3_y),(batch4_x, batch4_y) in loop:
            model.train()
            batch_x = torch.cat([batch1_x, batch2_x, batch3_x, batch4_x], dim=0)
            batch_y = torch.cat([batch1_y, batch2_y, batch3_y, batch4_y], dim=0)
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
                    for (batch1_x, batch1_y),(batch2_x, batch2_y),(batch3_x, batch3_y),(batch4_x, batch4_y) in zip(*val_loaders):
                        batch_x = torch.cat([batch1_x, batch2_x, batch3_x, batch4_x], dim=0)
                        batch_y = torch.cat([batch1_y, batch2_y, batch3_y, batch4_y], dim=0)
                        batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                        logits = model(batch_x)
                        loss = criterion(logits, batch_y.long())
                        val_loss += loss.item()
                        
                        # 计算准确率
                        preds = torch.argmax(logits, dim=1)
                        correct += (preds == batch_y).sum().item()
                        total += batch_y.size(0)
                
                avg_val_loss = val_loss / len(val_loaders[0])
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