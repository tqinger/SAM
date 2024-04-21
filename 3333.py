import torch

# 指定二维张量的形状，例如 4 行 5 列
shape = (4, 5)

# 随机生成一个二维张量，元素值为 0 或 1
edge_mask = torch.randint(2, size=shape)
print(edge_mask)

for i in range(0, edge_mask.size(0)):
    for j in range(1, edge_mask.size(1) - 1):
        if edge_mask[i][j] == 1 and edge_mask[i][j + 1] == 0:
            edge_mask[i][j] = 255
        elif edge_mask[i][j] == 1 and edge_mask[i][j - 1] == 0:
            edge_mask[i][j] = 255
        else:
            edge_mask[i][j] = 0
print(edge_mask)