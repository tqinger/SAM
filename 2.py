import torch

# 创建一个示例的二维tensor掩码图
mask_tensor = torch.tensor([[0, 1, 0, 1, 0],
                            [1, 0, 0, 1, 1],
                            [0, 1, 1, 0, 0]])

# 找到每行最左边和最右边的1的索引
left_indices = torch.argmax(mask_tensor == 1, dim=1)
right_indices = mask_tensor.size(1) - 1 - torch.argmax(mask_tensor.flip(dims=[1]) == 1, dim=1)

# 创建一个新的tensor，将除了每行最左边和最右边的1以外的1都变成0
result_tensor = torch.zeros_like(mask_tensor)
for i in range(mask_tensor.size(0)):
    result_tensor[i][left_indices[i]] = 1
    result_tensor[i][right_indices[i]] = 1

print("原始的二维tensor掩码图:")
print(mask_tensor)
print("\n处理后的结果:")
print(result_tensor)
