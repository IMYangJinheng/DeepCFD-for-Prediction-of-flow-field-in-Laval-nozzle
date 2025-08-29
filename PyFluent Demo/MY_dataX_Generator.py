import pandas as pd
import numpy as np
import os
import json
import pickle

# 定义输入文件夹和输出文件名
input_folder = "MY_Output_Data"
output_file = "dataX.json"

# 存储所有文件的数据，文件名格式为"压强,温度.txt"
all_files_data = []
for f in os.listdir(input_folder):
    if f.endswith(".txt"):
        pressure, temperature = map(float, os.path.splitext(f)[0].split(','))
        all_files_data.append((pressure, temperature))

# 从CSV文件中加载喷管的坐标
file_path = 'Nozzle.csv'
nozzle_data = pd.read_csv(file_path)

# 将 'x' 和 'y' 列转换为数值型数据，将任何错误转换为NaN
nozzle_data['x'] = pd.to_numeric(nozzle_data['x'], errors='coerce')
nozzle_data['y'] = pd.to_numeric(nozzle_data['y'], errors='coerce')

# 删除任何包含NaN值的行以清理数据
nozzle_data_clean = nozzle_data.dropna()

# 定义笛卡尔平面的网格大小
grid_width = 390
grid_height = 100

# 创建一个填充为-1的空网格，代表喷管的外部区域
base_grid = np.full((grid_height, grid_width), -1)

# 从清理后的数据中提取x和y坐标
x_coords = nozzle_data_clean['x'].values
y_coords = nozzle_data_clean['y'].values

# 缩放x和y坐标以适应网格尺寸
x_scaled = (x_coords / x_coords.max() * (grid_width - 1)).astype(int)
y_scaled = (y_coords / y_coords.max() * (grid_height - 1)).astype(int)

all_grids = []

# 遍历所有数据并填充独立网格
for pressure, temperature in all_files_data:
    grid = base_grid.copy()  # 使用拷贝避免修改原始网格
    for i in range(len(x_scaled) - 1):
        x1, y1 = x_scaled[i], y_scaled[i]
        x2, y2 = x_scaled[i + 1], y_scaled[i + 1]

        for x in range(min(x1, x2), max(x1, x2) + 1):
            y_top = max(y1, y2)
            y_bottom = 0
            grid[y_bottom:y_top, x] = pressure  # 填充压强数据

    # 填充温度数据到对应的网格
    temperature_grid = base_grid.copy()  # 创建一个温度数据网格
    for i in range(len(x_scaled) - 1):
        x1, y1 = x_scaled[i], y_scaled[i]
        x2, y2 = x_scaled[i + 1], y_scaled[i + 1]

        for x in range(min(x1, x2), max(x1, x2) + 1):
            y_top = max(y1, y2)
            y_bottom = 0
            temperature_grid[y_bottom:y_top, x] = temperature  # 填充温度数据

    # 逆时针旋转 90 度
    grid_rotated = np.rot90(grid)  # 压强网格逆时针旋转 90 度
    temperature_grid_rotated = np.rot90(temperature_grid)  # 温度网格逆时针旋转 90 度

    # 将旋转后的压强和温度堆叠为一个新的样本
    all_grids.append(np.stack((grid_rotated, temperature_grid_rotated), axis=0))  # 压强和温度一起存储


# 自定义 JSON 格式化函数
def custom_json_formatter(data):
    result = []
    for file_data in data:
        # 使用 .astype(int) 将多维数组转为整数，并使用 .tolist() 转换为 Python 列表
        file_data_formatted = [category.astype(int).tolist() for category in file_data]
        result.append(file_data_formatted)
    return json.dumps(result, separators=(',', ':'))


# 保存为 JSON 文件
with open(output_file, 'w') as f:
    json_data = custom_json_formatter(all_grids)
    f.write(json_data)

# 定义输入 JSON 文件和输出 PKL 文件路径
json_file = "dataX.json"
pkl_file = "dataX.pkl"

# 读取 JSON 文件
with open(json_file, 'r') as f:
    data = json.load(f)

# 保存为 PKL 文件
with open(pkl_file, 'wb') as f:
    pickle.dump(data, f)

print(f"文件已成功从 {json_file} 转换为 {pkl_file}")
# 打印数据的形状
print("data shape:", np.array(data).shape)  # 打印数组的形状
