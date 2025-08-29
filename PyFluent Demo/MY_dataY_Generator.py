import os
import pandas as pd
from scipy.interpolate import griddata
import numpy as np
import pickle
import json

# 定义输入文件夹和输出文件名
input_folder = "MY_Output_Data"
output_file = "dataY.json"

# 存储所有文件的数据
all_files_data = []

# 读取喷管坐标数据
nozzle_data_path = 'Nozzle.csv'
nozzle_data = pd.read_csv(nozzle_data_path)

# 将 'x' 和 'y' 列转换为数值型数据，非数值的数据会被转化为 NaN
nozzle_data['x'] = pd.to_numeric(nozzle_data['x'], errors='coerce')
nozzle_data['y'] = pd.to_numeric(nozzle_data['y'], errors='coerce')

# 删除含有 NaN 值的行以清理数据
nozzle_data_clean = nozzle_data.dropna()

# 定义网格大小
grid_width = 390
grid_height = 100

# 从清理后的数据中提取 x 和 y 坐标
x_coords = nozzle_data_clean['x'].values
y_coords = nozzle_data_clean['y'].values

# 缩放 x 和 y 坐标以适应网格尺寸
x_scaled = (x_coords / x_coords.max() * (grid_width - 1)).astype(int)
y_scaled = (y_coords / y_coords.max() * (grid_height - 1)).astype(int)

# 创建一个填充为 -1 的空网格，代表喷管外部区域
grid = np.full((grid_height, grid_width), -1)

# 遍历所有 .txt 文件并填充数据
for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".txt"):
        # 读取 .txt 文件
        file_path = os.path.join(input_folder, filename)
        data = pd.read_csv(file_path, delim_whitespace=True)

        # 提取 x, y, velocity, temperature, pressure 数据
        x = data['x-coordinate']
        y = data['y-coordinate']
        velocity = data['mach-number']
        temperature = data['temperature']
        pressure = data['pressure']

        # 创建网格
        x_grid = np.linspace(x.min(), x.max(), grid_width)
        y_grid = np.linspace(y.min(), y.max(), grid_height)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

        # 插值
        points = np.column_stack((x, y))
        velocity_grid = griddata(points, velocity, (X_grid, Y_grid), method='linear', fill_value=0)
        temperature_grid = griddata(points, temperature, (X_grid, Y_grid), method='linear', fill_value=0)
        pressure_grid = griddata(points, pressure, (X_grid, Y_grid), method='linear', fill_value=0)

        # 创建一个掩码，只在喷管区域填充数据
        mask = np.zeros_like(grid, dtype=bool)

        # 标记喷管内部区域
        for i in range(len(x_scaled) - 1):
            x1, y1 = x_scaled[i], y_scaled[i]
            x2, y2 = x_scaled[i + 1], y_scaled[i + 1]

            for x in range(min(x1, x2), max(x1, x2) + 1):
                y_top = max(y1, y2)
                y_bottom = 0
                mask[y_bottom:y_top, x] = True  # 喷管内部区域为 True

        # 使用掩码进行填充，喷管区域使用插值结果，外部区域保持 -1
        velocity_grid = np.where(mask, velocity_grid, -1)
        temperature_grid = np.where(mask, temperature_grid, -1)
        pressure_grid = np.where(mask, pressure_grid, -1)

        # 将填充后的数据存储为嵌套列表
        velocity_columns = [list(column) for column in velocity_grid.T.round(3)]
        temperature_columns = [list(column) for column in temperature_grid.T.round(3)]
        pressure_columns = [list(column) for column in pressure_grid.T.round(3)]

        # 将速度、温度、压力分别存储为嵌套列表
        all_files_data.append([
            velocity_columns,  # 速度
            temperature_columns,  # 温度
            pressure_columns  # 压力
        ])

# 自定义 JSON 格式化函数，强制单行输出每列数据
def custom_json_formatter(data,i=0):
    result = "[\n"
    for file_data in data:
        result += "  [\n"
        for category in file_data:  # 速度、温度、压力
            result += "    [\n"
            for column in category:  # 每列数据
                result += f"      {json.dumps(column)},\n"  # 使用 json.dumps 保证正确格式
            result = result.rstrip(",\n") + "\n"  # 移除最后的逗号
            result += "    ],\n"
        result = result.rstrip(",\n") + "\n"  # 移除最后的逗号
        result += "  ],\n"
        i = i + 1
        print("正在处理，请耐心等待,已完成" + str(i) + '/500')
    result = result.rstrip(",\n") + "\n"  # 移除最后的逗号
    result += "]"
    return result

# 保存为自定义格式的 JSON 文件
with open(output_file, 'w') as f:
    f.write(custom_json_formatter(all_files_data))

print(f"数据处理完成，已保存到 {output_file}")

# 定义输入 JSON 文件和输出 PKL 文件路径
json_file = "dataY.json"
pkl_file = "dataY.pkl"

# 读取 JSON 文件
with open(json_file, 'r') as f:
    data = json.load(f)

# 保存为 PKL 文件
with open(pkl_file, 'wb') as f:
    pickle.dump(data, f)

print(f"文件已成功从 {json_file} 转换为 {pkl_file}")
# 打印数据的形状
print("data shape:", np.array(data).shape)  # 打印数组的形状
