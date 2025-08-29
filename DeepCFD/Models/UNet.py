import torch   #导入pytorch
import torch.nn as nn   #导入pytorch的神经网络模块
import torch.nn.functional as F   #导入pytorch的函数模块
from torch.nn.utils import weight_norm   #导入权重归一化工具，用于提升模型训练的稳定性
from Models.AutoEncoder import create_layer   #导入create_layer，用于创建网络层

#创建一个编码器，这是U-Net的一部分
'''
说明：编码器块通常由多个卷积层组成，每一层都在尝试提取输入数据的特征。
    以下是参数说明：
    in_channels (int): 编码器块的输入通道数。这是第一个卷积层的输入通道数，如果这个块是编码器的第一个块，那么它就是整个网络的输入通道数。
    out_channels (int): 编码器块的输出通道数。这是最后一个卷积层的输出通道数，通常用于下一个编码器块的输入。
    kernel_size (int or tuple): 卷积核的大小，可以是一个整数或者一个由两个整数组成的元组，表示卷积核的高和宽。
    wn (bool): 是否应用权重归一化（weight normalization）。这是一个可选的技术，用于提高训练的稳定性和速度。
    bn (bool): 是否应用批量归一化（batch normalization）。批量归一化是一种常用的技术，用于调整神经网络中间层的输出，目的是提高训练的稳定性和速度。
    activation (nn.Module): 激活函数，用于在卷积层之后引入非线性。默认使用的是 nn.ReLU，即修正线性单元。
    layers (int): 编码器块中卷积层的数量，默认为2层。
'''
def create_encoder_block(in_channels, out_channels, kernel_size, wn=True, bn=True,
                 activation=nn.ReLU, layers=2):

    encoder = []   #初始化一个空列表enconder，用于储存构成编码器的各个卷积层
    for i in range(layers):   #使用 for 循环创建指定数量的卷积层（由 layers 参数控制）
        _in = out_channels   #设定默认的输入通道数 _in 和输出通道数 _out 为 out_channels
        _out = out_channels
        if i == 0:   #检查当前层是否为编码器块的第一层（i == 0）。如果是，将输入通道数 _in 设为函数参数 in_channels
            _in = in_channels   #调用 create_layer 函数创建一个卷积层
        encoder.append(create_layer(_in, _out, kernel_size, wn, bn, activation, nn.Conv2d))   #调用 create_layer 函数创建一个卷积层
    return nn.Sequential(*encoder)   #使用 nn.Sequential 将 encoder 列表中的所有卷积层组合成一个顺序模型，并返回这个模型。
    # nn.Sequential 是 PyTorch 提供的一个容器，它按照它们在构造器中传递的顺序执行模块列表中的每个模块
#这个函数的目的是根据给定的参数创建一个具有多个卷积层的编码器块，这些卷积层可以捕获输入数据的特征，并将它们传递到 U-Net 架构中的下一个编码器块

#创造解码器块，这是U-Net中与编码器相对称的部分，以下参数与编码器模块相对应，不再解释
def create_decoder_block(in_channels, out_channels, kernel_size, wn=True, bn=True,
                 activation=nn.ReLU, layers=2, final_layer=False):
    decoder = []   #初始化一个空列表deconder，用于储存构成编码器的各个卷积层
    for i in range(layers):   #使用 for 循环创建指定数量的卷积层（由 layers 参数控制）
        _in = in_channels   #设定默认的输入通道数 _in 和输出通道数 _out 为 in_channels
        _out = in_channels
        _bn = bn
        _activation = activation
        if i == 0:   #检查当前层是否为解码器块的第一层（i == 0）。
            # 如果是，将输入通道数 _in 设为 in_channels * 2，因为解码器块的第一层会接收来自前一个解码器块的输出和对应编码器块的输出（通过跳跃连接）
            _in = in_channels * 2   #检查当前层是否为解码器块的最后一层（i == layers - 1）。
            # 如果是，将输出通道数 _out 设为 out_channels。如果这一层是解码器的最终层（final_layer 为 True），则不使用批量归一化和激活函数
        if i == layers - 1:
            _out = out_channels
            if final_layer:
                _bn = False
                _activation = None
        decoder.append(create_layer(_in, _out, kernel_size, wn, _bn, _activation, nn.ConvTranspose2d))
    return nn.Sequential(*decoder)   #使用 nn.Sequential 将 decoder 列表中的所有转置卷积层组合成一个顺序模型，并返回这个模型。
    # nn.Sequential 是 PyTorch 提供的一个容器，它按照它们在构造器中传递的顺序执行模块列表中的每个模块
#这个函数的目的是根据给定的参数创建一个具有多个转置卷积层的解码器块，这些转置卷积层可以逐步恢复输入数据的空间维度，并将它们传递到 U-Net 架构中的下一个解码器块

#创建完整的编码器，这个完整的编码器是由一个个小的encoder_block构成的
def create_encoder(in_channels, filters, kernel_size, wn=True, bn=True, activation=nn.ReLU, layers=2):
    encoder = []   #初始化一个空列表 encoder，用于存储构成整个编码器的各个编码器块
    for i in range(len(filters)):   #使用 for 循环遍历 filters 列表，为每个元素创建一个编码器块
        if i == 0:   #如果当前是列表中的第一个元素（即第一个编码器块），使用 in_channels作为输入通道数，当前元素值作为输出通道数，调用 create_encoder_block 函数创建第一个编码器块
            encoder_layer = create_encoder_block(in_channels, filters[i], kernel_size, wn, bn, activation, layers)
        else:   #对于后续的编码器块，使用前一个块的输出通道数作为当前块的输入通道数，当前元素值作为输出通道数，同样调用 create_encoder_block 函数创建编码器块
            encoder_layer = create_encoder_block(filters[i-1], filters[i], kernel_size, wn, bn, activation, layers)
        encoder = encoder + [encoder_layer]   #将每个创建的编码器块添加到 encoder 列表中
    return nn.Sequential(*encoder)   #使用 nn.Sequential 将 encoder 列表中的所有编码器块组合成一个顺序模型，并返回这个模型
#这个函数的目的是构建整个编码器部分，它通过多个卷积层逐步提取输入数据的特征并减少其空间维度。这对于后续的解码器部分来说是必要的，
# 因为解码器需要在这些特征的基础上重建或上采样数据，以进行图像分割、重建或其他任务。编码器的这种逐步降采样过程允许网络学习到输入数据的高级特征表示。

#创建完整的解码器，这个完整的编码器是由一个个小的decoder_block构成的
def create_decoder(out_channels, filters, kernel_size, wn=True, bn=True, activation=nn.ReLU, layers=2):
    decoder = []   ##初始化一个空列表 decoder，用于存储构成整个解码器的各个解码器块
    for i in range(len(filters)):   #使用 for 循环遍历 filters 列表，为每个元素创建一个编码器块
        if i == 0:   #如果当前是列表中的第一个元素（即第一个编码器块），使用 in_channels 作为输入通道数，当前元素值作为输出通道数，调用 create_encoder_block 函数创建第一个编码器块
            decoder_layer = create_decoder_block(filters[i], out_channels, kernel_size, wn, bn, activation, layers, final_layer=True)
        else:   #对于后续的编码器块，使用前一个块的输出通道数作为当前块的输入通道数，当前元素值作为输出通道数，同样调用 create_encoder_block 函数创建编码器块
            decoder_layer = create_decoder_block(filters[i], filters[i-1], kernel_size, wn, bn, activation, layers, final_layer=False)
        decoder = [decoder_layer] + decoder   #将每个创建的编码器块添加到 encoder 列表中
    return nn.Sequential(*decoder)   #使用 nn.Sequential 将 encoder 列表中的所有编码器块组合成一个顺序模型，并返回这个模型。
    # nn.Sequential 是 PyTorch 提供的一个容器，它按照它们在构造器中传递的顺序执行模块列表中的每个模块
#这个函数的目的是构建整个编码器部分，它通过多个卷积层逐步提取输入数据的特征并减少其空间维度。这对于后续的解码器部分来说是必要的，
# 因为解码器需要在这些特征的基础上重建或上采样数据，以进行图像分割、重建或其他任务。编码器的这种逐步降采样过程允许网络学习到输入数据的高级特征表示

'''
这段代码定义了一个名为 UNet 的类，它继承自 PyTorch 的 nn.Module 类。UNet 类实现了 U-Net 架构，该架构广泛用于图像分割任务。
U-Net 有一个编码器-解码器结构，通过跳跃连接将编码器的特征图与解码器的相应层连接起来。以下是对这个类及其方法的详细解释:

in_channels (int): 输入图像的通道数。
out_channels (int): 输出图像的通道数。
kernel_size (int, optional): 卷积核的大小，默认为 3。
filters (list, optional): 每个编码器块输出通道数的列表，默认为 [16, 32, 64]。
layers (int, optional): 每个编码器/解码器块中的卷积层数，默认为 2。
weight_norm (bool, optional): 是否在卷积层中使用权重归一化，默认为 True。
batch_norm (bool, optional): 是否在卷积层中使用批量归一化，默认为 True。
activation (callable, optional): 激活函数，默认为 nn.ReLU。
final_activation (callable, optional): 最终输出前的激活函数。
'''
class UNet(nn.Module):   #UNet 的类，它继承自 PyTorch 的 nn.Module 类
    def __init__(self, in_channels, out_channels, kernel_size=3, filters=[16, 32, 64], layers=2,
                 weight_norm=True, batch_norm=True, activation=nn.ReLU, final_activation=None):

        super().__init__()   #__init__: 类的构造函数，用于初始化网络。它创建编码器和解码器，并保存最终激活函数
        assert len(filters) > 0   # 确保 filters不为空
        self.final_activation = final_activation   # 保存最终激活函数
        self.encoder = create_encoder(in_channels, filters, kernel_size, weight_norm, batch_norm, activation, layers)   #创建编码器
        self.decoder = create_decoder(out_channels, filters, kernel_size, weight_norm, batch_norm, activation, layers)    #创建解码器
#enconde对输入数据 x 进行编码的方法。它通过编码器逐步降采样输入数据，同时保存中间层的特征图、最大池化操作的索引和大小。这些信息将用于解码过程
    def encode(self, x):   # 编码过程
        tensors = []   #保存编码器每层的输出特征
        indices = []   #indices 保存了最大池化操作的索引，以便后续进行反池化操作
        sizes = []   #保存了每层输出的空间维度
        for encoder in self.encoder:   ## 遍历解码器中的每个块
            x = encoder(x)   #应用编码器块
            sizes.append(x.size())   #保存当前层的输出大小
            tensors.append(x)   #保存当前层的输出
            x, ind = F.max_pool2d(x, 2, 2, return_indices=True)   #进行最大池化操作并返回索引
            # kernel_size=2 表示池化窗口的大小为 2x2，即每次在 2x2 的窗口内选取最大值。2x2 池化窗口意味着特征图的高度和宽度会缩小一半
            indices.append(ind)   #保存索引
        return x, tensors, indices, sizes   #返回编码器的最终输出和中间结果
#decode: 对编码后的数据 x 进行解码的方法。它使用编码器层的特征图和解码器层进行上采样，并通过跳跃连接将特征图与编码器层连接起来，以重建图像的精细结构
    def decode(self, x, tensors, indices, sizes):
        for decoder in self.decoder:   #遍历解码器中的每个块
            tensor = tensors.pop()   #获取编码器的对应输出
            size = sizes.pop()   #获取编码器输出的大小
            ind = indices.pop()   #获取最大池化操作的索引
            x = F.max_unpool2d(x, ind, 2, 2, output_size=size)   #F.max_unpool2d 用于执行最大池化操作的逆操作，即上采样
            x = torch.cat([tensor, x], dim=1)   #torch.cat 用于将编码器的特征图与解码器的特征图沿通道维度拼接起来
            x = decoder(x)   #应用解码器块
        return x   #返回解码器的最终输出
#forward: 定义模型的前向传播过程。它先调用 encode 方法进行编码，然后调用 decode 方法进行解码。如果定义了 final_activation，则在输出之前应用最终激活函数
    def forward(self, x):
        x, tensors, indices, sizes = self.encode(x)   #进行编码
        x = self.decode(x, tensors, indices, sizes)   #进行解码
        if self.final_activation is not None:   #如果定义了最终激活函数
            x = self.final_activation(x)   #应用最终激活函数
        return x
