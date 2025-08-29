import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 设置全局字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# 设置图片分辨率为 300 DPI（高清）
plt.rcParams["figure.dpi"] = 300

# 设置字体大小
plt.rcParams["font.size"] = 12

def split_tensors(*tensors, ratio):
    assert len(tensors) > 0
    split1, split2 = [], []
    count = len(tensors[0])
    for tensor in tensors:
        assert len(tensor) == count
        split1.append(tensor[:int(len(tensor) * ratio)])
        split2.append(tensor[int(len(tensor) * ratio):])
    if len(tensors) == 1:
        split1, split2 = split1[0], split2[0]
    return split1, split2

def initialize(model, gain=1, std=0.02):
    for module in model.modules():
        if type(module) in [nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d]:
            nn.init.xavier_normal_(module.weight, gain)
            if module.bias is not None:
                nn.init.normal_(module.bias, 0, std)

def visualize(sample_y, out_y, error, s, y_mean, y_std):
# def visualize(sample_y, out_y, error, s):
    # 将 Tensor 转换为 numpy 数组
    y_mean = y_mean.cpu().detach().numpy()  # 转换为 numpy 数组
    y_std = y_std.cpu().detach().numpy()    # 转换为 numpy 数组

    ###########################################
    # 对 sample_y 和 out_y 进行逆标准化
    sample_y_denorm = sample_y * y_std + y_mean
    out_y_denorm = out_y * y_std + y_mean
    error_denorm = error * y_std  # 误差通常与输出的尺度一致
    # sample_y_denorm = sample_y
    # out_y_denorm = out_y
    # error_denorm = error
    # # ###########################################



    # 获取样本的最小值和最大值
    minu = np.min(sample_y_denorm[s, 0, :, :])
    maxu = np.max(sample_y_denorm[s, 0, :, :])

    minv = np.min(sample_y_denorm[s, 1, :, :])
    maxv = np.max(sample_y_denorm[s, 1, :, :])

    minp = np.min(sample_y_denorm[s, 2, :, :])
    maxp = np.max(sample_y_denorm[s, 2, :, :])

    # 误差的最小值和最大值
    mineu = np.min(error_denorm[s, 0, :, :])
    maxeu = np.max(error_denorm[s, 0, :, :])

    minev = np.min(error_denorm[s, 1, :, :])
    maxev = np.max(error_denorm[s, 1, :, :])

    minep = np.min(error_denorm[s, 2, :, :])
    maxep = np.max(error_denorm[s, 2, :, :])

    # 绘图
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(15, 10)

    # 绘制速度图
    plt.subplot(3, 3, 1)
    plt.title('CFD', fontsize=18)
    plt.imshow(np.transpose(sample_y_denorm[s, 0, :, :]), cmap='jet', vmin=minu, vmax=maxu, origin='lower', extent=[0, 390, 0, 100])
    plt.colorbar(orientation='horizontal')
    plt.ylabel('Velocity', fontsize=18)

    # 绘制 U-Net 输出速度图
    plt.subplot(3, 3, 2)
    plt.title('U-Net', fontsize=18)
    plt.imshow(np.transpose(out_y_denorm[s, 0, :, :]), cmap='jet', vmin=minu, vmax=maxu, origin='lower', extent=[0, 390, 0, 100])
    plt.colorbar(orientation='horizontal')

    # 绘制误差图
    plt.subplot(3, 3, 3)
    plt.title('Error', fontsize=18)
    plt.imshow(np.transpose(error_denorm[s, 0, :, :]), cmap='jet', vmin=mineu, vmax=maxeu, origin='lower', extent=[0, 390, 0, 100])
    plt.colorbar(orientation='horizontal')

    # 绘制温度图
    plt.subplot(3, 3, 4)
    plt.imshow(np.transpose(sample_y_denorm[s, 1, :, :]), cmap='jet', vmin=minv, vmax=maxv, origin='lower', extent=[0, 390, 0, 100])
    plt.colorbar(orientation='horizontal')
    plt.ylabel('Temperature', fontsize=18)

    # 绘制 U-Net 输出温度图
    plt.subplot(3, 3, 5)
    plt.imshow(np.transpose(out_y_denorm[s, 1, :, :]), cmap='jet', vmin=minv, vmax=maxv, origin='lower', extent=[0, 390, 0, 100])
    plt.colorbar(orientation='horizontal')

    # 绘制温度误差图
    plt.subplot(3, 3, 6)
    plt.imshow(np.transpose(error_denorm[s, 1, :, :]), cmap='jet', vmin=minev, vmax=maxev, origin='lower', extent=[0, 390, 0, 100])
    plt.colorbar(orientation='horizontal')

    # 绘制压力图
    plt.subplot(3, 3, 7)
    plt.imshow(np.transpose(sample_y_denorm[s, 2, :, :]), cmap='jet', vmin=minp, vmax=maxp, origin='lower', extent=[0, 390, 0, 100])
    plt.colorbar(orientation='horizontal')
    plt.ylabel('Pressure', fontsize=18)

    # 绘制 U-Net 输出压力图
    plt.subplot(3, 3, 8)
    plt.imshow(np.transpose(out_y_denorm[s, 2, :, :]), cmap='jet', vmin=minp, vmax=maxp, origin='lower', extent=[0, 390, 0, 100])
    plt.colorbar(orientation='horizontal')

    # 绘制压力误差图
    plt.subplot(3, 3, 9)
    plt.imshow(np.transpose(error_denorm[s, 2, :, :]), cmap='jet', vmin=minep, vmax=maxep, origin='lower', extent=[0, 390, 0, 100])
    plt.colorbar(orientation='horizontal')

    plt.tight_layout()

    # 保存为高清图片
    plt.savefig(f"flow_field_{s}.png", bbox_inches="tight", dpi=300)
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extract_wall_boundary_numpy(field, threshold=1e-6):
    """
    对每一列，从上往下找第一个大于 threshold 的值的索引
    """
    wall_y = []
    height, width = field.shape
    for x in range(width):
        col = field[:, x]
        # 从上往下（即从 y=0 到 y=height-1）
        for y in range(height):
            if col[y] > threshold:
                wall_y.append(y)
                break
        else:
            wall_y.append(None)  # 整列都是 0
    return wall_y

def extract_wall_and_plot_numpy(error_denorm, s):
    """
    提取壁面误差，绘图并保存为 CSV（error_denorm 是 NumPy 格式）
    输入 shape: [batch_size, 3, height, width]
    """
    var_names = ['Velocity', 'Temperature', 'Pressure']

    for i, name in enumerate(var_names):
        try:
            field = np.transpose(error_denorm[s, i, :, :])  # 转为 [height, width]
        except Exception as e:
            print(f"[错误] 第 {i} 个变量 {name} 转置失败：{e}")
            continue

        wall_y = extract_wall_boundary_numpy(field)

        # 去除 None 值
        valid_indices = [i for i, y in enumerate(wall_y) if y is not None]
        valid_wall_y = [wall_y[i] for i in valid_indices]

        if not valid_wall_y:
            print(f"[警告] 在 {name} 中未检测到壁面")
            continue

        # 保存 CSV
        df = pd.DataFrame({
            'x': valid_indices,
            'wall_y': valid_wall_y
        })
        df.to_csv(f'wall_boundary_error_{name.lower()}.csv', index=False)

        # 绘图
        plt.figure(figsize=(8, 4))
        plt.imshow(field, cmap='jet', origin='lower', extent=[0, 390, 0, 100])
        x_real = np.linspace(0, 390, len(wall_y))
        y_real = [y * 100 / field.shape[0] if y is not None else np.nan for y in wall_y]

        plt.plot(x_real, y_real, color='white', linewidth=1.5, label='Wall Contour')
        plt.title(f"{name} Error with Wall Detection", fontsize=14)
        plt.colorbar(label='Error Magnitude')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"error_wall_{name.lower()}.png", dpi=300)
        plt.show()


