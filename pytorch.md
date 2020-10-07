# 常见函数

## 生成随机数tensor

<u>**torch.manual_seed()**</u>

- `torch.manual_seed(args.seed)` #为CPU设置种子用于生成随机数，以使得结果是确定的
  `if args.cuda:`
  	`torch.cuda.manual_seed(args.seed)` #为当前GPU设置随机种子；如果使用多个GPU，应该使用``torch.cuda.manual_seed_all()``为所有的GPU设置种子。

- 不同的初始化参数往往会导致不同的结果，通过设置随机数种子使结果可复现

<u>**torch.rand(*sizes, out=None)**</u>

​	**均匀分布**

​	返回一个张量，包含了从区间[0, 1)的均匀分布中抽取的一组随机数。张量的形状由参数sizes定义

```python
torch.rand(2, 3)
# 0.0836 0.6151 0.6958
# 0.6998 0.2560 0.0139
```

<u>**torch.randn(*sizes, out=None)**</u>

​	**标准正态分布**

​	返回一个张量，包含了从标准正态分布（均值为0，方差为1，即高斯白噪声）中抽取的一组随机数。张量的形状由参数sizes定义。

```python
torch.randn(2, 3)
# 0.5419 0.1594 -0.0413
# -2.7937 0.9534 0.4561
```

**<u>torch.normal(means, std, out=None)</u>**

​	**离散正态分布**	

​	返回一个张量，包含了从指定均值means和标准差std的离散正态分布中抽取的一组随机数。

- means (float, optional) - 均值
- std (Tensor) - 标准差
- out (Tensor) - 输出张量

```python
torch.normal(mean=0.5, std=torch.arange(1, 6))
-0.1505
-1.2949
-4.4880
-0.5697
-0.8996
```

**<u>torch.arange()</u>**

​	torch.arange(start, end, step=1, out=None) → Tensor

​	返回1维张量，包含从`start`到`end`，以`step`为步长的一组序列值(默认步长为1)。

```python
torch.arange(1, 4)
'''
1
2
3
[torch.FloatTensor of size 3]
'''
```

## 运算

### 加法

```python
print(torch.add(x, y))
y.add_(x) # in-place
```

### 获取值

```python
# 一个元素张量可以用x.item()得到元素值
print(x.item())
```

### 改变矩阵形状

#### torch.Tensor.view()

不改变张量数据的情况下随意改变张量的大小和形状。类似于Numpy的`np.reshape()`

返回的张量共享相同的数据。如果对"view"中进行更改，则需要更改原始张量数据

```python
x = torch.randn(4, 4)
z = x.view(-1, 8)  # -1表示该位从其他维度中推算
print z.size()
# (2,8)
```

#### torch.transpose(*input*, *dim0*, *dim1*)

- 交换给定维度 `dim0` and `dim1`

  - 只能操作2D矩阵

- input (Tensor) – 输入张量，必填

  dim0 (int) – 转置的第一维，默认0，可选

  dim1 (int) – 转置的第二维，默认1，可选

- ```python
  >>> x = torch.randn(2, 3)
   tensor([[ 1.0028, -0.9893,  0.5809],
          [-0.1669,  0.7299,  0.4942]])
  >>> torch.transpose(x, 0, 1)
  tensor([[ 1.0028, -0.1669],
          [-0.9893,  0.7299],
          [ 0.5809,  0.4942]])   
  ```

#### permute(dims)

- ```python3
  torch.Tensor.permute
  ```

- 将tensor的维度换位

#### squeeze

- 对数据的维度进行压缩或者解压。
- 去掉维数为1的维度

#### unsqueeze

- 对数据维度进行扩充
- 指定位置加上维数为一的维度

## 神经网络

### 模型参数

一个模型可训练的参数可以通过调用 net.parameters() 返回：

```python
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight
```

输出：

```python
10
torch.Size([6, 1, 5, 5])
```

### ImageFolder

```python
from torchvision.datasets import ImageFolder
```

**ImageFolder**(`root`,`transform`=None,`target_transform`=None,`loader`=default_loader)
	`root` : 在指定的root路径下面寻找图片
	`transform`: 对PIL Image进行转换操作,transform 输入是PIL image，输出transform后的图像
	`target_transform `:takes in the target and transforms it 

​	`loader`: A function to load an image given its path 加载指定图片

### Conv2d

```python
torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
                stride=1, padding=0, dilation=1, groups=1, bias=True)
```

- 二维卷积层
- 输入尺度是(N, C_in,H,W)，输出尺度（N,C_out,H_out,W_out）
- 参数
  - in_channels(`int`) – 输入图像的通道数
  - out_channels(`int`) – 卷积产生的通道数
  - kerner_size(`int` or `tuple`) - 卷积核（filter）的尺寸
  - stride(`int` or `tuple`, `optional`) - 卷积步长
  - padding (`int` or `tuple`, `optional`)- 填充圈数
  - dilation(`int` or `tuple`, `optional``) – 卷积核元素之间的间距
  - groups(`int`, `optional`) – 从输入通道到输出通道的阻塞连接数
    - 控制输入和输出之间的连接： `group=1`，输出是所有的输入的卷积；`group=2`，此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来。
  - bias(`bool`, `optional`) - 如果`bias=True`，添加偏置

### ConvTranspose2d

```python
ConvTranspose2d(in_channels, out_channels, kernel_size,
                stride=1, padding=0, output_padding=0, groups=1, bias=True)
```

- 二维反卷积层（可视为解卷积，但不是真正的解卷积操作），可以视为**计算Conv2d的梯度**
- 将一张m * m 的图像放大为 n *ｎ（ｎ＞ｍ）
- 由于内核的大小，输入的最后的一些列的数据可能会丢失，用户可以进行适当的填充（`padding`操作）。
- 参数
  - in_channels(`int`) – 输入图像的通道数
  - out_channels(`int`) – 卷积产生的通道数
  - kerner_size(`int` or `tuple`) - 卷积核（filter）的尺寸
  - stride(`int` or `tuple`, `optional`) - 卷积步长
  - padding(`int` or `tuple`, `optional`) - 输入填充圈数（0）
  - output_padding(`int` or `tuple`, `optional`) - 输出的填充圈数（0）
  - dilation(`int` or `tuple`, `optional`) – 卷积核元素之间的间距
  - groups(`int`, `optional`) – 从输入通道到输出通道的阻塞连接数
  - bias(`bool`, `optional`) - 如果`bias=True`，添加偏置

### Linear



# pytorch基础

## CUDA

device = torch.device(“cuda” if torch.cuda.is_available() else “[cpu](https://www.baidu.com/s?wd=cpu&tn=24004469_oem_dg&rsv_dl=gh_pl_sl_csd)”)，

会依据你的计算机配置自动选择CPU还是[GPU](https://www.baidu.com/s?wd=GPU&tn=24004469_oem_dg&rsv_dl=gh_pl_sl_csd)运算。

```python
# 定义device，是否使用GPU，依据计算机配置自动会选择
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 用.to(device)来决定Tensor or Module使用GPU还是CPU
  model = CNN().to(device)

# 训练
  total_loss = 0
  for input, target in train_loader:
      #不需要转化为Variable，直接用Tensors作为输入，用.to(device)来决定使用GPU还是CPU
      input, target = input.to(device), target.to(device)
      hidden = input.new_zeros(*h_shape)  
     ''' ...'''  
# 获得loss值，也与老版本不同
      total_loss += loss.item()          
  # 测试
  with torch.no_grad():  # 测试时不会进行梯度计算，节约内存
      for input, target in test_loader:
          '''...'''
```

##Tensor

**Tensor**，张量，表示多维的矩阵，pytorch处理的对象。torch.Tensor能够使用GPU进行加速。

0维的Tensor就是**scalar**

```python
# 未初始化矩阵
x = torch.Tensor(5, 3)
# 随机矩阵
x = torch.rand(5, 3)

print('x.size()：'，x.size())
# out：('x.size(): ', torch.Size([5, 3]))

# 指定生成的Tensor的数据类型
# dtype = torch.FloatTensor CPU
dtype = torch.cuda.FloatTensor
x = torch.rand(5, 3).type(dtype) # GPU

print('x.type():',x.type())
# out: ('x.type(): ', 'torch.cuda.FloatTensor')
```

**Tensor与Numpy互转换**

```python
x_numpy = x_Tensor.numpy()  # Tensor -> numpy
y_Tensor = torch.from_numpy(y_numpy)  # numpy -> Tensor
# 转换后，两者的内存位置共享,in-place运算使两者值均改变
x_tensor.add_(1)
# 如果没有使用in-place运算，则无法共享内存
y_numpy = y_numpy + 1
# in-place运算,类似'+='运算，此时直接赋值给y,使用`_`作为后缀
y.add_(x)

y.cuda()  # 转化为GPU
```

##Variable （现与Tensor 合并）

- **Variable**,提供自动求导的功能，自动给你想要的参数的梯度。　
- 原`.creator`由`.grad_fn`取代
- 原`.requires_grad`成为了Tensor的属性

```python
import torch

x = torch.ones(1)
x.requires_grad_(True)
# requires_grad表示在backward是否计算其梯度
y = torch.ones(1).requires_grad_(True)
z = 2 * x + y +4
z.backward()
print('dz/dx:{}'.format(x.grad.data))
print('dz/dy:{}'.format(y.grad.data))
result:
    dz/dx:
        2 #对x求得的导数
        [torch.FloatTensor of size 1]
    dz/dy:
        1
        [torch.FloatTensor of size 1]
```

### 自动微分

- 将`torch.Tensor`的`.required_grad`设置为`True`
- 通过调用`.backward`自动计算所有梯度，该张量的梯度将累积到 `.grad `属性中
- [leaf tensor](https://blog.csdn.net/wangweiwells/article/details/101223420)
  - 反向传播中只计算叶节点且`requires_grad=True`的梯度
    - `requires_grad=False`反向传播中无意义节点
  - 可以理解成不依赖其他tensor的tensor
  - 神经网络层中的权值w的tensor为叶子节点；自己定义的tensor是叶子节点
  - [辅助阅读1](https://zhuanlan.zhihu.com/p/85506092)

##  状态词典

- state_dict

  - 字典对象

  - 将每一层与它的对应参数建立映射关系

  - 生成

    - 定义了model或optimizer之后pytorch自动生成的,可以直接调用，保存state_dict的格式是".pt"或'.pth'的文件

  - 保存

    - ```python
      torch.save(model.state_dict(), PATH)
      ```

  - 加载

    - ```python
      model = TheModelClass(*args, **kwargs)
      model.load_state_dict(torch.load(PATH))
      model.eval()
      ```

      - model.eval() 很重要
      - "dropout层"及"batch normalization层"在"训练(training)模态"与"评估(evalution)模态"下有不同的表现形式

    - 加载某一层的state

      - ```python
        conv1_weight_state = torch.load('./model_state_dict.pt')['conv1.weight']
        ```

- [阅读](https://www.cnblogs.com/marsggbo/p/12075356.html)
- 

## `torch.nn`和`torch.nn.functional`

```python
import torch.nn as nn
import torch.nn.functional as F
# 基本的网络构建类模板
class net_name(nn.Module):
    def _init_(self):
        super(net_name,self)._init_()
        # 可以添加各种网络层
        self.convl = nn.Conv2d(3,10,3)
        # 具体每种层的参数查找文档
    def forward(self,x):
        # 定义向前传播
        out = self.convl(x)
        return out
```

### 简介

pytorch集成的现成模块，将网络高度模块化。包括常见的网络层，如卷积、池化，RNN计算以及Loss计算等。可以把torch.nn包内的各个类想象成神经网络的一层，把torch.nn以及torch.nn.Modules看做是一个定义好的小网络（包括其参数），给该网络一个输入，其将得到一个输出。输入输出都是Variable。

### 常见函数(列举一些)

| torch.nn                                                     | 用法                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| class torch.nn.Sequential(* args)                            | 时序容器。`Modules` 会以他们传入的顺序被添加到容器中，在forward时按顺序进行操作，如下即Conv->BN->ReLU->Pool |
| zero_grad()                                                  | 将`module`中的所有模型参数的梯度设置为0.                     |
| class torch.nn.ReLU(inplace=False)                           | 对输入运用修正线性单元函数${ReLU}(x)= max(0, x)$             |
| **class** **torch**.**nn**.**Linear**(in_features, out_features, bias=True) | 对输入数据做线性变换：y=Ax+b                                 |

```python
# Example of using Sequential

model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
# Example of Relu()
m = nn.ReLU()
input = autograd.Variable(torch.randn(2))
print(m(input))

```

### 机制

- torch.nn包里面的大部分类都是继承了父类`torch.nn.Modules`

- `torch.nn`（类）和`torch.nn.functional`（函数）功能很接近，类可维护性高，函数比较灵活

- 区别：

  - nn.Xxx 需要**先实例化并传入参数**，然后以函数调用的方式调用实例化的对象并传入输入数据。

    ```python
    inputs = torch.rand(64, 3, 244, 244)
    
    conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
    
    out = conv(inputs)
    ```

  - nn.functional.xxx**同时传入**<u>输入数据</u>和<u>weight, bias等其他参数</u> 

    ```python
    weight = torch.rand(64,3,3,3)
    bias = torch.rand(64) 
    
    out = nn.functional.conv2d(inputs, weight, bias, padding=1)
    ```

  - `nn.Xxx`不需要你自己定义和管理weight；

    `nn.functional.xxx`需要每次调用时手动传入weight

### 注意点

具有学习参数的（例如，conv2d, linear, batch_norm)采用`nn.Xxx`方式；

没有学习参数的（例如，maxpool, loss func, activation func）等根据个人选择使用`nn.functional.xxx`或者`nn.Xxx`方式。

但关于**dropout**，个人强烈推荐使用`nn.Xxx`方式，因为一般情况下只有训练阶段才进行dropout，在eval阶段都不会进行dropout。使用`nn.Xxx`方式定义dropout，在调用`model.eval()`之后，model中所有的dropout layer都关闭，

## torch.optim

pytorch集成的模块，包括许多常见的优化算法 SGD 、Adam等

```python
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameter(), lr = learning_rate)

# 将对应参数梯度置0,注意如果没有置0，则将会与上一次梯度进行叠加
optimizer.zero_grad()
# forward 然后 backward 得到梯度
y = model(x)
loss = loss_fn(y_pred, y)
optimizer.step() # 更新参数
```

`optimizer.step()`

​	一旦梯度被如`backward()`之类的函数计算好后，我们就可以调用这个函数,这个方法会更新所有的参数

## 自定义数据读取

　　使用`torch.utils.data.Dataset`和`torch.utils.data.dataloader`组合为数据迭代器。每次训练时，利用这个迭代器输出每一个batch数据，并在输出时对数据进行相应的预处理或数据增广操作。

　　`torchvision`继承`torch.utils.data.Dataset`

　　自定义数据读取方式要继承`torch.utils.data.Dataset`并将其封装到`DataLoader`中。`Dataset`表示该数据集，实现多种**数据读取**和**预处理**方式。`DataLoader`封装了Data对象，实现单（多）进程迭代器**输出数据集**

### torch.utils.data.Dataset

　　至少重载两个方法，`__len__`，`__getitem__`

- `__len__`返回数据集大小
- `__getitem__`实现索引数据集中的某一个数据
  - 还可以在`__getitem__`时对数据进行预处理，或直接在硬盘中读取数据，对于超大的数据集可以使用`lmdb`

```python
import torch
from torch.utils.data import DataLoader,Dataset
class TensorDatase(Dataset):
    """通过index得到数据集的数据，能够通过len，得到数据集大小"""
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]
    def __len__(self):
        return self.data_tensor.size(0)

data_tensor = torch.randn(4, 3)
target_tensor = torch.rand(4)
tensor_dataset = TensorDatase(data_tensor, target_tensor)
print('tensor_data[0]:',tensor_dataset[0])
print('len', len(tensor_dataset))
```

### torch.utils.data.DataLoader

　　`Dataloader`将Dataset或其子类封装成一个迭代器。这个迭代器可以迭代输出Dataset的内容，同时可以实现多进程、shuffle、不同的采样策略，数据校对等处理过程。

```python
tensor_dataloader = DataLoader(tensor_dataset,
                               batch_size=2,
                               shuffle=True,  # 随机输出
                               num_workers=0)  # 只有1个进程
for data, target in tensor_dataloader:
    print(data, target)

print('one batch tensor data:', iter(tensor_dataloader).next())
print('len of batchtensor:', len(list(iter(tensor_dataloader))))
```

### torchvision.datasets

　　`torchvision`包包括许多常用的CNN模型以及一些数据集。`torchvision.dataset`包含了MINIST，cifar10等数据集，他们都继承了上述的Dataset类。

#### 调用自带数据集

```python
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
show = ToPILImage()       # 把Tensor转成Image，方便可视化
import matplotlib.pyplot as plt
import torchvision
import numpy as np
###############数据加载与预处理
transform = transforms.Compose([transforms.ToTensor(),  # 转为tensor
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),#归一化
                                ])
#训练集
trainset=tv.datasets.CIFAR10(root='../data',
                             train=True,
                             download=True,
                             transform=transform)

trainloader=t.utils.data.DataLoader(trainset,
                                    batch_size=4,
                                    shuffle=True,
                                    num_workers=0)
classes = ('plane','car','bird','cat','deer','dog','frog','horse','ship','truck')
(data, label) = trainset[100]
print(classes[label])
show((data+1)/2).resize((100,100))

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(images.size())
imshow(torchvision.utils.make_grid(images))
plt.show()
```

#### 硬盘载入图像

`torch.datasets`包中的`ImageFolder`支持我们直接从硬盘中按照固定路径格式载入每张数据`根目录/类别/图像`

## 预处理

+ `torchvision.transforms`

+ 实现各种**预处理**

+ PIL.Image/numpy.ndarray与Tensor的**相互转化**；

  + 常用于训练模型阶段的数据读取（`PIL.image`或`numpy.ndarray` -> Tensor）

    + `transforms.ToTensor()`

      将`PIL.image`或`numpy.ndarray`数据转化为`torch.FloadTensor`，并归一化到[0,1.0]：

      - 取值范围为[0, 255]的PIL.Image，转换成形状为[C, H, W]，取值范围是[0, 1.0]的torch.FloadTensor；
      - 形状为[H, W, C]的numpy.ndarray，转换成形状为[C, H, W]，取值范围是[0, 1.0]的torch.FloadTensor。

  + 用于验证模型阶段的数据输出（Tensor ->` PIL.image`或`numpy.ndarray`）

    + `transforms.ToPILImage()`

      将Tensor转化为PIL.Image

    + `.numpy()`

      将Tensor转化为numpy

  ```python
  transform1 = transforms.Compose([
      transform.ToTensor(),
  ])
  # numpy.ndarray
  img = cv2.imread(img_path)# 读取图像
  img1 = transform1(img) # 归一化到 [0.0,1.0]
  print("img1 = ",img1)
  # 转化为numpy.ndarray并显示
  img_1 = img1.numpy()*255
  img_1 = img_1.astype('uint8')
  img_1 = np.transpose(img_1, (1,2,0))
  cv2.imshow('img_1', img_1)
  cv2.waitKey()
  #PIL
  img = Image.open(image_path).convert('RGB') #读取图像
  img2 = transform1(img)
  print('img2 = ',img2)
  #PILImage
  img_2 = transforms.ToPILImage()(img2).convert('RGB')
  print("img_2 = ", img_2)
  img_2.show()
  ```

+ 归一化；

  ```python
  transform2 = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(mean = (0.5,0.5,0.5),std = (0.5,0.5,0.5))
  ])
  ```

  使用如下公式进行归一化：channel=（channel-mean）/std

+ 对PIL.Image进行裁剪、缩放等操作。

+ 使用Compose将这些预处理方式组成transforms list，对图像进行多种处理 

## 采样

### 上采样（upsampling）

- 扩大图像

- 又称为图像插值法
- 在原有图像像素的基础上在像素点之间采用合适的插值算法插入新的元素

###下采样(subsampled)

- 缩小图像

- 对于一幅尺寸为`M*N`图像I，对其进行s倍下采样，即得到`(M/s)*(N/s)`尺寸的得分辨率图像

# torch常用库

## Numpy

## SciPy

**简介**

SciPy 包含的模块有最优化、线性代数、积分、插值、特殊函数、快速傅里叶变换、信号处理和图像处理、常微分方程求解和其他科学与工程中常用的计算。

## Matplotlib

**简介**

Matplotlib 是 Python 编程语言及其数值数学扩展包 NumPy 的可视化操作界面。它为利用通用的图形用户界面工具包，如 Tkinter, wxPython, Qt 或 GTK+ 向应用程序嵌入式绘图提供了应用程序接口（API）。

# 国内镜像源

http://mirrors.ustc.edu.cn/anaconda/pkgs/free/

https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/

https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/

https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/