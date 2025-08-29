from curses.ascii import ispunct
import numpy as np
import pickle   #导入pickle库，用于加载预处理模型
import torch   #导入pytorch，用于深度学习模型的构建

# def calculate_isp(temperature_tru, temperature_pre, pressure_e_pre, pressure_e_tru, pressure_0_pre, pressure_0_tru, gama = 1.4, kuozhangbi = 5, pressure_a = 101325, R = 8.314):
def calculate_isp_tru(y):
    gama = 1.4
    kuozhangbi = 5
    pressure_a = 101325
    R = 287
    # pai_e_pre = pressure_e_pre/pressure_0_pre
    # pai_a_pre = pressure_a/pressure_0_pre
    isp_tru_cell = []

    for i in range(y.shape[0]):
        # 提取第 i 个样本
        sample = y[i]

        # 提取温度通道（通道1）第一行的最大值作为 T
        temperature_tru = sample[1, 0, :].max().item()

        # 提取压强通道（通道2）第一行的最大值作为 P0
        pressure_0_tru_in = sample[2, 0, :].max().item()

        # 提取压强通道（通道2）最后一行的最大值作为 Pe
        pressure_e_tru_in = sample[2, -1, :].max().item()

        pressure_e_tru = pressure_e_tru_in + pressure_a
        pressure_0_tru = pressure_0_tru_in + pressure_a

        # print(f"得到的出口处压强Pe: {pressure_e_tru}")
        # print(f"得到的燃烧室压强P0: {pressure_0_tru}")

        pai_e_tru = pressure_e_tru / pressure_0_tru
        pai_a_tru = pressure_a / pressure_0_tru

        isp_tru = (R * temperature_tru) ** (0.5) * (
                ((2 * gama / (gama - 1) * (1 - pai_e_tru ** ((gama - 1) / gama))) ** 0.5) + (kuozhangbi ** 2 / gama) * (
                pai_e_tru - pai_a_tru))
        # isp_pre = (R * temperature_pre) ** (0.5) * (((2 * gama / (gama - 1) * (1 - pai_e_pre ** ((gama - 1) / gama))) ** 0.5) + (kuozhangbi ** 2 / gama) * (pai_e_pre - pai_a_pre))

        # 存储结果
        isp_tru_cell.append(isp_tru)

    print(f"计算得到的比冲数量: {len(isp_tru_cell)}")
    print(f"前5个比冲值: {isp_tru_cell[:-5]}")
