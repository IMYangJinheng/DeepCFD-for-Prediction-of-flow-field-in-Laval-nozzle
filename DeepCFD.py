import os   #导入os库，用于操作文件和目录
import json   #导入json库，用于处理JSON数据格式
import torch   #导入pytorch，用于深度学习模型的构建
import pickle   #导入pickle库，用于加载预处理模型
from train_functions import *   #导入之前写好的train_function
from functions import *   #导入之前写好的function
import torch.optim as optim   #导入pytorch的优化器
from torch.utils.data import TensorDataset   #导入tensordataset，用于创建数据集
from Models.UNetEx import UNetEx   #导入之前写好的UNetEx
# from Models.AutoEncoder import AutoEncoder   #导入之前写好的UNetEx
from other_Data import *
import matplotlib.pyplot as plt

# 设置全局字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# 设置图片分辨率为 300 DPI（高清）
plt.rcParams["figure.dpi"] = 300

# 设置字体大小
plt.rcParams["font.size"] = 12

if __name__ == "__main__":   #主函数入口

    # 加载数据集
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 判断是否GPU可用，若没有则使用CPU
    x = pickle.load(open("./basic_dataset/dataX.pkl", "rb"))  # 加载数据集dataX，rb指打开格式read＆binary（二进制），即以二进制格式打开数据集dataX
    y = pickle.load(open("./basic_dataset/dataY.pkl", "rb"))  # 同上

    # 检查数据类型并转换为 PyTorch 张量
    x = torch.FloatTensor(x)  # 将dataX转换为张量
    y = torch.FloatTensor(y)  # 将dataY转换为张量

    # 确保dataX的形状是[500, 1, 390, 100]，通过unsqueeze添加通道维度
    x = x.unsqueeze(1)  # 添加通道维度
    print("Shape of dataX after unsqueeze:", x.shape)  # 应该是[500, 1, 390, 100]

    # 打印dataY的形状确认
    print("Shape of dataY:", y.shape)  # 应该是[500, 3, 390, 100]

    ##############################
    # 标准化数据
    x_mean = x.mean()
    x_std = x.std()
    x = (x - x_mean) / x_std

    y_mean = y.mean(dim=(0, 2, 3), keepdim=True)
    y_std = y.std(dim=(0, 2, 3), keepdim=True)
    y = (y - y_mean) / y_std
    ###############################

    # 计算每个通道的权重，确保channels_weights是形状为 (1, 3, 1, 1)
    channels_weights = torch.sqrt(torch.mean(y.permute(0, 2, 3, 1).reshape((500 * 390 * 100, 3)) ** 2, dim=0)).view(1,-1,1,1).to(device)
    print("Channels weights:", channels_weights)  # 打印通道权重，用于后续的损失函数中平衡不同通道的误差

    # 训练模型和其他设置...

    simulation_directory = "./Run/"   #设置结果的保存目录为.run
    if not os.path.exists(simulation_directory):
        os.makedirs(simulation_directory)   #如果run目录不存在，那么就创建这个目录

    # 切分数据集为训练集和测试集
    train_data, test_data = split_tensors(x, y, ratio=0.8)    # 将数据集划分为训练集七，测试集三，这里用的是split_tensors函数进行的切割

    train_dataset, test_dataset = TensorDataset(*train_data), TensorDataset(*test_data)
    '''
    创建两个数据集，train_dataset用于训练，test_dataset用于测试。
    使用TensorDataset进行的分组，将原来三七开的数据集进行分组
    '''
    test_x, test_y = test_dataset[:]   #将测试集再进行切片，一个叫test_x，一个叫test_y

    torch.manual_seed(0)   #设置随机种子，以保证实验的可重复性
    '''
    为了保证实验的可重复，将随机种子设置为0。这意味着每次运行代码时，所有随机操作（如权重初始化、数据洗牌等）都会产生相同的结果
    如果不固定随机种子，每次运行代码时，这些随机操作的结果可能会不同，从而导致实验结果的波动。
    通过设置固定的随机种子，可以确保每次运行代码时，这些随机操作的结果都是相同的，从而使得实验结果具有可比性和可重复性
    语法：torch.manual_seed(seed):   torch.manual_seed 是PyTorch中用于设置全局随机种子的函数。（seed 是一个整数，表示随机数生成器的初始值。）
    '''
    lr = 0.001   #学习率
    kernel_size = 5   #卷积核
    filters = [8, 16, 32, 32]   #设置卷积层的卷积核数量。第一个卷积层有8个过滤器，第二个卷积层有16个过滤器，依此类推。
    bn = False   #不在卷积层之后使用批量归一化
    wn = False   #不在模型中使用权重归一化
    wd = 0.005   #设置权重衰减，用于正则化

    # 创建模型实例，in_channels=1，out_channels=3
    model = UNetEx(in_channels=1, out_channels=3, filters=filters, kernel_size=kernel_size, batch_norm=bn, weight_norm=wn)
    # model = AutoEncoder(in_channels=1, out_channels=3, filters=filters, kernel_size=kernel_size, batch_norm=bn,weight_norm=wn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)   #定义优化器，用的是AdamW优化器
    '''
    了解一下：AdamW是一种流行的优化算法，它结合了动量（Momentum）和自适应学习率的优点。
    AdamW与标准Adam优化器的主要区别在于它是如何应用权重衰减的：
    标准Adam：权重衰减是直接加在损失函数上，然后对总损失进行梯度下降。
    AdamW：权重衰减是作为权重更新的一部分，而不是损失函数的一部分。这意味着权重衰减不会影响梯度，而是直接作用在参数更新上。
    '''

    config = {}   #创建一个字典，用来储存训练的结果
    train_loss_curve = []   #用于存储训练损失曲线
    test_loss_curve = []   #用于存储测试损失曲线
    train_mse_curve = []   #用于存储训练MSE曲线
    test_mse_curve = []   #用于存储测试MSE曲线
    train_ux_curve = []   #用于存储训练Ux MSE曲线
    test_ux_curve = []   #用于存储测试Ux MSE曲线
    train_uy_curve = []   #用于存储训练Uy MSE曲线
    test_uy_curve = []   #用于存储测试Uy MSE曲线
    train_p_curve = []   #用于存储训练压力MSE曲线
    test_p_curve = []   #用于存储测试压力MSE曲线

    #####################
    train_uxape_curve = []
    test_uxape_curve = []
    train_uyape_curve = []
    test_uyape_curve = []
    train_pape_curve = []
    test_pape_curve = []
    isp_tru = []
    isp_pre = []
    #####################

    def after_epoch(scope):   #定义一个函数，咋每个训练周期结束后执行一次
        train_loss_curve.append(scope["train_loss"])  # 记录训练损失
        test_loss_curve.append(scope["val_loss"])  # 记录测试损失
        train_mse_curve.append(scope["train_metrics"]["mse"])  #记录训练MSE
        test_mse_curve.append(scope["val_metrics"]["mse"])  #记录测试MSE
        train_ux_curve.append(scope["train_metrics"]["ux"])  #记录训练Ux MSE
        test_ux_curve.append(scope["val_metrics"]["ux"])  #记录测试Ux MSE
        train_uy_curve.append(scope["train_metrics"]["uy"])  #记录训练Uy MSE
        test_uy_curve.append(scope["val_metrics"]["uy"])  #记录测试Uy MSE
        train_p_curve.append(scope["train_metrics"]["p"])  #记录训练压力MSE
        test_p_curve.append(scope["val_metrics"]["p"])   #记录测试压力MSE

        ##########################
        # 新增APE和PMAE的记录
        train_uxape_curve.append(scope["train_metrics"]["uxape"])
        test_uxape_curve.append(scope["val_metrics"]["uxape"])

        train_uyape_curve.append(scope["train_metrics"]["uyape"])
        test_uyape_curve.append(scope["val_metrics"]["uyape"])

        train_pape_curve.append(scope["train_metrics"]["pape"])
        test_pape_curve.append(scope["val_metrics"]["pape"])
        ##########################

    def loss_func(model, batch):
        x, y = batch
        output = model(x)
        lossu = ((output[:,0,:,:] - y[:,0,:,:]) ** 2).reshape((output.shape[0],1,output.shape[2],output.shape[3]))
        lossv = ((output[:,1,:,:] - y[:,1,:,:]) ** 2).reshape((output.shape[0],1,output.shape[2],output.shape[3]))
        lossp = torch.abs((output[:,2,:,:] - y[:,2,:,:])).reshape((output.shape[0],1,output.shape[2],output.shape[3]))
        loss = (lossu + lossv + lossp)/channels_weights
        return torch.sum(loss), output

    # Training model
    DeepCFD, train_metrics, train_loss, test_metrics, test_loss = train_model(model, loss_func, train_dataset, test_dataset, optimizer,
        epochs=1000, batch_size=64, device=device,
        m_mse_name="Total MSE",
        m_mse_on_batch=lambda scope: float(torch.sum((scope["output"] - scope["batch"][1]) ** 2)),
                                                                              m_mse_on_epoch=lambda scope: sum(
                                                                                  scope["list"]) / len(
                                                                                  scope["dataset"]),
                                                                              m_ux_name="Velocity MSE",
                                                                              m_ux_on_batch=lambda scope: float(
                                                                                  torch.sum((scope["output"][:, 0, :,
                                                                                             :] - scope["batch"][1][:,
                                                                                                  0, :, :]) ** 2)),
                                                                              m_ux_on_epoch=lambda scope: sum(
                                                                                  scope["list"]) / len(
                                                                                  scope["dataset"]),
                                                                              m_uy_name="Temperature MSE",
                                                                              m_uy_on_batch=lambda scope: float(
                                                                                  torch.sum((scope["output"][:, 1, :,
                                                                                             :] - scope["batch"][1][:,
                                                                                                  1, :, :]) ** 2)),
                                                                              m_uy_on_epoch=lambda scope: sum(
                                                                                  scope["list"]) / len(
                                                                                  scope["dataset"]),
                                                                              m_p_name="Presure MSE",
                                                                              m_p_on_batch=lambda scope: float(
                                                                                  torch.sum((scope["output"][:, 2, :,
                                                                                             :] - scope["batch"][1][:,
                                                                                                  2, :, :]) ** 2)),
                                                                              m_p_on_epoch=lambda scope: sum(
                                                                                  scope["list"]) / len(
                                                                                  scope["dataset"]),
                                                                              ############
                                                                              # 新增通道级APE指标（与MSE的思路类似）
                                                                              m_uxape_name="Ux APE",
                                                                              m_uxape_on_batch=lambda scope: float(
                                                                                  100.0 * torch.sum(torch.abs(
                                                                                  scope["batch"][1][:, 0, :, :] - scope["output"][:, 0, :, :]))
                                                                                  / torch.sum(torch.abs(
                                                                                      scope["batch"][1][:, 0, :, :]))
                                                                              ),
                                                                              m_uxape_on_epoch=lambda scope: sum(
                                                                                  scope["list"]) / len(
                                                                                  scope["dataset"]),

                                                                              m_uyape_name="Uy APE",
                                                                              m_uyape_on_batch=lambda scope: float(
                                                                                  100.0 * torch.sum(torch.abs(
                                                                                  scope["batch"][1][:, 1, :, :] - scope["output"][:, 1, :, :]))
                                                                                  / torch.sum(torch.abs(
                                                                                      scope["batch"][1][:, 1, :, :]))
                                                                              ),
                                                                              m_uyape_on_epoch=lambda scope: sum(
                                                                                  scope["list"]) / len(
                                                                                  scope["dataset"]),

                                                                              m_pape_name="P APE",
                                                                              m_pape_on_batch=lambda scope: float(
                                                                                  100.0 * torch.sum(torch.abs(
                                                                                  scope["batch"][1][:, 2, :, :] - scope["output"][:, 2, :, :]))
                                                                                  / torch.sum(torch.abs(
                                                                                      scope["batch"][1][:, 2, :, :]))
                                                                              ),
                                                                              m_pape_on_epoch=lambda scope: sum(
                                                                                  scope["list"]) / len(
                                                                                  scope["dataset"]),

                                                                              ##########
                                                                              patience=25, after_epoch=after_epoch
                                                                              )

    # 保存训练过程中的各种曲线数据
    metrics = {}
    metrics["train_metrics"] = train_metrics
    metrics["train_loss"] = train_loss
    metrics["test_metrics"] = test_metrics
    metrics["test_loss"] = test_loss
    curves = {}
    curves["train_loss_curve"] = train_loss_curve
    curves["test_loss_curve"] = test_loss_curve
    curves["train_mse_curve"] = train_mse_curve
    curves["test_mse_curve"] = test_mse_curve
    curves["train_ux_curve"] = train_ux_curve
    curves["test_ux_curve"] = test_ux_curve
    curves["train_uy_curve"] = train_uy_curve
    curves["test_uy_curve"] = test_uy_curve
    curves["train_p_curve"] = train_p_curve
    curves["test_p_curve"] = test_p_curve
    config["metrics"] = metrics
    config["curves"] = curves
    # 假设列表是数字类型的误差数据
    results = {
        "train_uxape_curve": train_uxape_curve,
        "test_uxape_curve": test_uxape_curve,
        "train_uyape_curve": train_uyape_curve,
        "test_uyape_curve": test_uyape_curve,
        "train_pape_curve": train_pape_curve,
        "test_pape_curve": test_pape_curve
    }

    print("即将保存APE结果到ape_curves.json")
    with open("ape_curves.json", "w") as f:
        json.dump(results, f)
    print("保存成功！")

    # 生成测试结果
    out = DeepCFD(test_x[:10].to(device))  # 测试前10个数据
    error = torch.abs(out.cpu() - test_y[:10].cpu())  # 计算误差
    calculate_isp_tru(out)
    s = 0  # 设置可视化的序列索引
    visualize(test_y[:10].cpu().detach().numpy(), out[:10].cpu().detach().numpy(), error[:10].cpu().detach().numpy(),
              s, y_std, y_mean)  # 可视化测试结果
    # ###########################################
    # visualize(test_y[:10].cpu().detach().numpy(), out[:10].cpu().detach().numpy(), error[:10].cpu().detach().numpy(),
    #           s)  # 可视化测试结果
    # ###########################################
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss_curve, label="Train Loss")
    plt.plot(test_loss_curve, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.show()

    ####################
    results = {
        "train_uxape_curve": train_uxape_curve,
        "test_uxape_curve": test_uxape_curve,
        "train_uyape_curve": train_uyape_curve,
        "test_uyape_curve": test_uyape_curve,
        "train_pape_curve": train_pape_curve,
        "test_pape_curve": test_pape_curve,
    }

    with open("metrics_no_results.json", "w") as f:
        json.dump(results, f)
    ####################
    plt.figure()
    plt.plot(train_uxape_curve, label="Train Velocity APE")
    plt.plot(test_uxape_curve, label="Validation Velocity APE")
    plt.xlabel("Epoch")
    plt.ylabel("APE (%)")
    plt.title("Train vs Validation Velocity APE")
    plt.legend()
    plt.show()

    # 对 Uy APE
    plt.figure()
    plt.plot(train_uyape_curve, label="Train Temperature APE")
    plt.plot(test_uyape_curve, label="Validation Temperature APE")
    plt.xlabel("Epoch")
    plt.ylabel("APE (%)")
    plt.title("Train vs Validation Temperature APE")
    plt.legend()
    plt.show()

    # 对 P APE
    plt.figure()
    plt.plot(train_pape_curve, label="Train Pressure APE")
    plt.plot(test_pape_curve, label="Validation Pressure APE")
    plt.xlabel("Epoch")
    plt.ylabel("APE (%)")
    plt.title("Train vs Validation Pressure APE")
    plt.legend()
    plt.show()

    # 将数据存储为字典
    print(type(train_uxape_curve))

    data = {
        "train_uxape": train_uxape_curve,  # 直接使用列表
        "test_uxape": test_uxape_curve,
        "train_uyape": train_uyape_curve,
        "test_uyape": test_uyape_curve,
        "train_pape": train_pape_curve,
        "test_pape": test_pape_curve
    }

    # 保存为 JSON 文件
    with open("data_without_gs", "w") as f:
        json.dump(data, f)
    ####################

    print("正在保存模型...请耐心等候")
    # 保存模型
    model_save_path = os.path.join( "./model_complete.pth")
    print(f"正在保存模型到：{model_save_path}")
    try:
        torch.save(model.state_dict(), model_save_path)
        print("模型已保存！")
    except Exception as e:
        print(f"保存模型时发生错误: {e}")

    error = torch.abs(out.cpu() - test_y[:10].cpu()).detach().numpy()
    extract_wall_and_plot_numpy(error, s)
    print("误差结果已保存")




