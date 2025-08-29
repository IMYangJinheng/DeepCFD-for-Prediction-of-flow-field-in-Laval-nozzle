import os
import json
import torch
import pickle
from torch.utils.data import TensorDataset
from Models.UNetEx import UNetEx
from train_functions import *

if __name__ == "__main__":
    # 加载数据集
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = pickle.load(open("./dataX.pkl", "rb"))
    y = pickle.load(open("./dataY.pkl", "rb"))
    x = torch.FloatTensor(x).to(device)
    y = torch.FloatTensor(y).to(device)

    # 检查是否保存过模型，如果有则加载模型，否则训练并保存
    model_path = './model.pth'

    if os.path.exists(model_path):
        # 加载保存的模型权重
        model = UNetEx(3, 3, filters=[8, 16, 32], kernel_size=5)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        print("模型已加载！")
    else:
        # 训练模型
        model = UNetEx(3, 3, filters=[8, 16, 32], kernel_size=5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.005)
        train_data, test_data = split_tensors(x, y, ratio=0.7)
        train_dataset, test_dataset = TensorDataset(*train_data), TensorDataset(*test_data)

        # 训练代码，省略...

        # 训练完成后保存模型
        torch.save(model.state_dict(), model_path)
        print("模型已保存！")

    # 进行预测
    out = model(x[:10])
    error = torch.abs(out.cpu() - y[:10].cpu())
    visualize(y[:10].cpu().detach().numpy(), out.cpu().detach().numpy(), error.cpu().detach().numpy(), 0)
