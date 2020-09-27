# 数组（arrays）

- numpy 封装
- 任意维度的矩阵
- 索引从0开始

## 创建

- 参数既可以是list，也可以是元组

  - ```python
    a=np.array((1,2,3,4,5))# 参数是元组
    b=np.array([1,2,3,4,5])# 参数是list
    print(a.shape)	#数组大小（n,m）nxm
    print(a.dtype)	#元素类型
    print(type(a))  # <class 'numpy.ndarray'>
    ```

- axis

  - ```python
    a = np.array([[0,1,1,1],[0,2,1,0],[1,0,0,1],[1,0,0,0]])
    # list 行向量
    a.sum(axis=0) #按列
    Out[8]: array([2, 3, 2, 2])
    a.sum(axis=1) #按行
    Out[9]: array([3, 3, 2, 1])
        
    a = np.array([[1,2],[3,4]])
    # 按行相加，并且保持其二维特性
    print(np.sum(a, axis=1, keepdims=True))
    # 按行相加，不保持其二维特性
    print(np.sum(a, axis=1))
    array([[3], [7]])
    array([3, 7])
    ```

## 转换

- List转numpy.array:

```python
temp = np.array(list) 
```

- numpy.array转List:

```python
arr = temp.tolist() 
```

## 数组索引

切片、整数值索引、布尔值索引

```python
import numpy as np
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(a[:2,1:3])	#切片
'''
[[2,3]
[6,7]]
'''
print(a[0,0],a[1,1])	#整数值索引
bool_idx = a > 6
print(bool_idx)
'''
[[False F F F]
 [F F T T]
 [T T T T]]
'''
print(a[bool_idx])	#整数值索引
#[7 8 9 10 11 12]
```

# 数学计算

线性代数

```python
import numpy as np
x = np.array([[1,2],[3,4]],dtype=np.float64)
y = np.array([[5,6],[7,8]],dtype=np.float64)
print(x)
print(y)
print(x+y)		#对应元素处理
print(x*y)
print(x/y)
print(np.dot(x,y))	#矩阵乘法
'''
[[1. 2.]
 [3. 4.]]
[[5. 6.]
 [7. 8.]]
[[ 6.  8.]
 [10. 12.]]
[[ 5. 12.]
 [21. 32.]]
[[0.2        0.33333333]
 [0.42857143 0.5       ]]
[[19. 22.]
 [43. 50.]]
'''
```

# 常用函数

```python
#运算
x = np.array([[1,2],[3,4]]) #[1,2]为一行，[1,3]为一列 
#[[1,2]
# [3,4]]
print(np.sum(x))	#求和 10
print(np.sum(x,axis=0))	# [4,6] 纵向相加
print(np.sum(x,axis=1))	# [3,7] 横向相加

a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print (a.shape)
#(3,4)

#创建指定矩阵
a = np.zeros((2,2))	#创建全为0的矩阵（2*2）
a = np.ones((1,2)) #创建全为1的矩阵（1*2）
a = np.full((2,2), 7)	#创建全为7的矩阵 （2*2）
a = np.eye(2)	#创建秩为n的单位矩阵（2）
a = np.eye(3,2,dtype=int);	#创建指定大小的对角矩阵
'''
[[1. 0.]
 [0. 1.]
 [0. 0.]]
 '''
a = np.random.random((2,2))	#创建指定大小的随机数矩阵 （2*2）

c = np.array([1,2,3])
d = c.reshape(3,1)
print(d,d.shape)
print(np.dot(a,d))
'''
[[1],
 [2],
 [3]] (3, 1)
[[14]
 [32]
 [50]]
'''

#求矩阵的逆，转置
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
b = np.mat(a)
print(b,'\n',b.T,'\n',b.I)	#原，转置，逆
# mat()函数作用
#将目标数据的类型转换为矩阵（matrix）

#求均值，标准差
>>> np.mean(a) # 对所有元素求均值
>>> np.mean(a,0) # 压缩行，对各列求均值
>>> np.std(a) # 计算全局标准差
>>> np.std(a, axis=0) # 计算每一列的标准差
# 标准差是方差的算术平方根
# 方差：所有数减去其平均值的平方和，所得结果除以该组数之个数
```

## np.random

- **np.random.rand(d0,d,...,dn)**
  - rand函数根据给定维度生成[0,1)之间的数据，包含0，不包含1
  - dn表示维度
  - 返回值为指定维度的array（数组）

- **np.random.randn(d0,d,...,dn)**

从**标准正态分布**中返回一个或多个样本值。

 ![pytorch_1](picformd\pytorch_1.png)

- np.random.normal(size,loc,scale):

从**正态（高斯）分布**中抽取随机样本。

给出输出**规模（数量）为size，均值为loc，标准差为scale**的高斯随机数

```python
>>> mu, sigma = 0, 0.1 # mean and standard deviation
>>> s = np.random.normal(mu, sigma, 4)
# [ 0.12414249 -0.0081143  -0.00109165  0.14299805]
```

## np.transpose()

作用：改变序列

对于二维 ndarray，transpose在不指定参数是默认是矩阵转置。

[解释]: https://blog.csdn.net/u012762410/article/details/78912667	"CSDN"

第一个方括号是0轴，第二个是1轴，以此类推

`np.transpose(npimg, (1, 2, 0))`

意思是交换npimg的`1轴 -> 0轴 2轴 -> 1轴  0轴 -> 2轴`

```python
A = np.arange(16)
A = a.reshape(2,2,4)
'''
A = array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7]],

           [[ 8,  9, 10, 11],
            [12, 13, 14, 15]]])
'''
A.transpose((0,1,2)) # 保持不变
A.transpose((1,0,2)) # 交换0轴和1轴
"""
A = array([[[ 0,  1,  2,  3],
			[ 8,  9, 10, 11]],
		   [[ 4,  5,  6,  7],
			[12, 13, 14, 15]]])
"""
```

## 转换数组中Nan和Inf

- **numpy.nan_to_num(x)**:

  - 使用0代替数组x中的`nan`元素，使用有限的数字代替`inf`元素

- ```python
  >>> a array([[  nan,   inf],
               [  nan,  -inf]]) 
  >>> np.nan_to_num(a) array([[ 0.00000000e+000,  1.79769313e+308],
                              [ 0.00000000e+000, -1.79769313e+308]])
  ```

- 和此类问题相关的还有一组判断用函数，包括：
  - isinf
  - isneginf
  - isposinf
  - isnan
  - isfinite