import torch
import os
from DataGenerator import FakeList, MapRouteDataset

def generate_uniform_jump_data(total_len=10000, save_path="data/", do_std=False):
    """
    生成 total_len 条数据，其中 min_jump 在 [1,2,3,4] 上均匀分布（各占 1/4）。
    返回两个列表，每个元素是一个 batch（形状为 [1, ...]）：
        - inputs: 列表，每个元素形状 (1, input_dim)
        - labels: 列表，每个元素形状 (1,)
    """
    assert total_len % 4 == 0, "total_len must be multiple of 4"
    per_class = total_len // 4

    M, L = 10, 10
    seed_bias = [5,6,7,8]  # 每个 min_jump 使用不同种子，保证数据不重复

    all_inputs = []
    all_labels = []

    for jump, seed in zip([1, 2, 3, 4], seed_bias):
        print(f"Generating {per_class} samples with min_jump={jump} ...")
        fakelist = FakeList(M, L, per_class, seed, min_jump=jump)
        dataset = MapRouteDataset(M, L, fakelist, do_std=do_std)

        for idx in range(len(dataset)):
            map_tensor, route_tensor = dataset[idx]
            # 添加 batch 维度，形状变为 [1, ...]
            all_inputs.append(map_tensor.unsqueeze(0))
            all_labels.append(route_tensor.unsqueeze(0))

    print(f"Generated {len(all_inputs)} samples.")
    return all_inputs, all_labels


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    inputs, labels = generate_uniform_jump_data(total_len=10000, do_std=False)

    torch.save(inputs, "eval_data/inputs.pt")
    torch.save(labels, "eval_data/labels.pt")
    print("Data saved to data/inputs.pt and data/labels.pt")
    print(f"Inputs list length: {len(inputs)}, first element shape: {inputs[0].shape}")
    print(f"Labels list length: {len(labels)}, first element shape: {labels[0].shape}")