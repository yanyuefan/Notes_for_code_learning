# type hint

- [阅读一](https://sikasjc.github.io/2018/07/14/type-hint-in-python/)
- 

#  基础知识

## 标识符：

- 第一个字符必须是字母表中字母或下划线 _ 。
- 标识符的其他的部分由字母、数字和下划线组成。
- 大小写敏感。

## 注释：

```python
# 第一个注释
print ("Hello, Python!") # 第二个注释
'''
第三注释
第四注释
'''
"""
第五注释
第六注释
"""
```

## 缩进：

​	缩进表示代码块，同一行代码块的语句必须包含相同的缩进空格数

## 多行语句：

　　如果语句很长，可以使用**反斜杠实现多行语句**

```python
total = item_one + \
        item_two + \
        item_three
```

​	在 [], {}, 或 () 中的多行语句，不需要使用反斜杠(\)

```python
total = ['item_one', 'item_two', 'item_three',
        'item_four', 'item_five']
```

## 数据类型：

​	python变量不需要声明，每个变量在使用前必须赋值；Python 中变量就是变量，它没有类型，我们所说的"类型"是变量所指的内存中对象的类型。Python允许你同时为多个变量赋值

 ![python_1](F:\机器学习笔记\工具笔记\picformd\python_1.png)

### 标准数据类型：

​	不可变数据：Number（数字）、String（字符串）、Tuple（元祖）

​	可变数据：List（列表）、Set（集合）、Dictionary（字典）

### Number

​		int（整数）  bool（布尔） float（浮点型、1.23、3e2（300））

​		complex（复数、1+2j、1.1+2.2j）

​	当指定一个值时，Number对象就会被创建：var1 = 1

​	删除对象引用：del var1[,val2[,var3[….,valN]]]

#### 数值运算

  ![python_2](F:\机器学习笔记\工具笔记\picformd\python_2.png)

#### 字符串

```python
word = '字符串'
sentence = "这是一个句子。"
paragraph = "这是一个段落，\
可以由多行组成"
```

​	Python 中的字符串有两种索引方式，从左往右以 0 开始，从右往左以 -1 开始。

​	Python中的字符串不能改变。

​	字符串的截取的语法格式如下：**变量[头下标:尾下标:步长]**

 ![python_3](F:\机器学习笔记\工具笔记\picformd\python_3.png)

![python_4](F:\机器学习笔记\工具笔记\picformd\python_4.png)

python中的转义符号‘\’

![python_4](F:\机器学习笔记\工具笔记\picformd\python_4_1.jpg)

```python
print("我叫 %s 今年 %d 岁！"%('小明',10))
"""
%c 格式化字符
%s 字符串
%d 整数
%f 浮点数
%e 科学计数法格式化浮点数
%p 十六进制变量的地址

```

#### 元祖

元祖类似列表，但无法改变。

元祖可以作为字典里的key，列表不可以。

```python
d = {(x, x+1): x for x in range(10)}	#利用元祖构建一个字典
t = (5, 6)	#构建一个元祖
print(t in d)	#true
print(d[t])	#5
print(d[(1, 2)])	#1
```

#### List

列表中元素的**类型可以不相同**，它支持数字，字符串甚至可以包含列表

列表是写在**方括号 [ ]**之间、用逗号分隔开的元素列表

列表被截取后返回一个包含所需元素的新列表：变量[头下标 : 尾下标]

 ![python_5](F:\机器学习笔记\工具笔记\picformd\python_5.png)

加号 **+** 是列表**连接**运算符，星号 ***是重复**操作

- [添加元素](http://smilejay.com/2013/02/add-items-to-a-list-in-python/)

  - ```python
    l1
    Out[46]: [[1, 1, 1], [2, 2, 2]]
    l2
    Out[47]: [[3, 3, 3]]
    l1+l2
    Out[48]: [[1, 1, 1], [2, 2, 2], [3, 3, 3]]
    ```

    

#### 集合

集合内的元素没有顺序关系，里面也**没有重复**的元素，写在{ }之间，逗号隔开

函数

```python
set()  # 构建空集合
anis = {'a','b'}	#构建一个集合
print('a' in anis)	#验证该元素是否在集合中
anis.add('c')	#为集合增加元素
print(len(anis))	#输出集合的长度
anis.remove('c')	#从集合中移除元素
print(len(anis))

set([iterable])     # 创建一个无序不重复元素集
```

#### 字典

字典里面存的是数据对，以(key,value)的形式存在且映射。

```python
dict = {}  # 构建空字典
d = {'cat': 'cute', 'dog': 'furry'}	#构建一个字典
print(d['cat'])	#打印字典中的元素
print('fish' in d)	#验证该元素是否在字典中
d['fish'] = 'wet'	#增加字典元素
print(d.get('fish', 'N/A'))	#wet
print(d.get('fis', 'N/A'))	#N/A
del d['dog']	#删除字典中的元素
print(d)	#输出字典
```

### 参数

#### 参数传递

- 关键字参数

  - 传入0个或任意个含参数名的参数，这些关键字参数在函数内部自动组装成一个`dict`
  - ![img](https://img-blog.csdn.net/20150901141835215?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

- 可变参数

  - 传入的参数个数可变：0、1、2、3...
  - 调用一个list或者tuple，在list或tuple前面加上一个\*号，把list或tuple的元素变成可变参数传进去

- 参数定义的顺序：必选参数、默认参数、可变参数和关键字参数

- 默认参数一定要用不可变对象，如果是可变对象，运行会有逻辑错误！

  `*args`是可变参数，`args`接收的是一个`tuple`;

  `**kw`是关键字参数，kw接收的是一个`dict`



## 常用操作

- 空行作用：分隔两段不同功能或含义的代码，便于日后代码的维护或重构
- 等待用户输入：

```python
#!/usr/bin/python3
input("\n\n click enter key and then the program exit.")
```

 ![python_6](F:\机器学习笔记\工具笔记\picformd\python_6.png)

- Python可以在同一行中使用多条语句，语句之间使用分号(;)分割
- 多个语句构成代码组（缩进相同的一组语句）


像if、while、def和class这样的**复合语句**，首行**以关键字开始，以冒号(: )结束**，该行之后的一行或多行代码构成代码组。我们将首行及后面的代码组称为一个子句(clause)。

 ![python_7](F:\机器学习笔记\工具笔记\picformd\python_7.png)

- print输出


print输出默认换行，如果不换行在变量末尾加end=""

![python_8](F:\机器学习笔记\工具笔记\picformd\python_8.png)

- in ：在指定的序列中找到值返回 True，否则返回 False

  not in：在指定的序列中没有找到值返回 True，否则返回 False

  is：两个标识符是引用自**同一个**对象 **id(x) == id(y)**

  is not：两个标识符不是引用自不同对象 **id(a) != id(b)**

- 模块导入


在 python 用 import 或者 from...import 来导入相应的模块。

导入整个模块(some module)： `import somemodule`

从某个模块中导入某个函数： `from somemodule import somefunction`

从某个模块中导入多个函数,格式为：`from somemodule import firstfunc, secondfunc, thirdfunc`

将某个模块中的全部函数导入，格式为： `from somemodule import *`

 ![python_9](F:\机器学习笔记\工具笔记\picformd\python_9.png)

## 分支

```python
if condition_1:
    statement_block_1
elif condition_2:
    statement_block_2
else:
    statement_block_3
"""
缩进划分语句块，相同缩进数的语句在一起组成一个语句块。
Python中没有switch – case语句
"""
```

## 循环

for循环 对列表中元素：从列表的开始遍历到列表的结束

```python
anis = ['cat','dog','monkey']
for ani in anis:
	print(ani)
    
sites = ["Baidu", "Google","Runoob","Taobao"]
for site in sites:
    if site == "Runoob":
        print("菜鸟教程!")
        break
    print("循环数据 " + site)
else:
    print("没有循环数据!")
print("完成循环!")
```

```python
#取得每个元素的下标，enumerate()
anis = ['cat','dog','monkey']
for idx,ani in enumerate(anis):
    print('#{}:{}'.format(idx + 1,ani))
"""
输出为
#1:cat
#2:dog
#3:monkey
"""
```

```python
nums = [0,1,2,3,4]
squares = [x**2 for x in nums]
#较高级的循环方式可以避免创建一个空列表进行元素添加
```

```python
# while语句
count = 0
while count < 5:
   print (count, " 小于 5")
   count = count + 1
else:
   print (count, " 大于或等于 5")
#  while … else 在条件语句为 false 时执行 else 的语句块
```

break 语句可以跳出 for 和 while 的循环体。如果你从 for 或 while 循环中终止，任何对应的循环 else 块将不执行

continue语句被用来告诉Python跳过当前循环块中的剩余语句，然后继续进行下一轮循环

pass是空语句，不做任何事情，一般用做占位语句

## 函数

通过 def 来定义

```python
def sign(x):
    if x > 0:
        return 'positive'
    elif x < 0:
        return 'negative'
    else:
        return 'zero'

    
for i in [-1, 0, 1]:
    print(sign(i))
```

不定长参数：处理比声明时更多的参数，声明时不会命名

```python
# 基本语法
def functionname([formal_args,]*var_args_tuple):
    "函数_文档字符串"
    function_suite
    return [expression]

# 加了星号 * 的参数会以元组(tuple)的形式导入，存放所有未命名的变量参数
def printinfo( arg1, *vartuple ):
   "打印任何传入的参数"
   print ("输出: ")
   print (arg1)
   print (vartuple)
 
# 调用printinfo 函数
printinfo( 70, 60, 50 )
# 70
# (60, 50)
```

```python
# 带 ** 基本语法
def functionname([formal_args,] **var_args_dict ):
   "函数_文档字符串"
   function_suite
   return [expression]

# 加了两个星号 ** 的参数会以字典的形式导入
def printinfo( arg1, **vardict ):
   "打印任何传入的参数"
   print ("输出: ")
   print (arg1)
   print (vardict)
 
# 调用printinfo 函数
printinfo(1, a=2,b=3)
# 1
# {'a':2,'b':3}
```


​	

## 类

类是抽象的模板，实例是根据类创造出来的具体对象

```python
class Student(object):

    def __init__(self, name, score):	#初始化
        self.name = name
        self.score = score

    def print_score(self):	#定义函数
        print('{}:{}'.format(self.name, self.score))


a = Student('a', 0)
b = Student('b', 99)
a.print_score()
b.print_score()
```

## 返回

- 两个值

  - 默认返回Tuple

  - ```python
    def read_tsv_data0(path: str) -> Tuple[pd.Series, pd.Series]:
    ```

    

# python常见函数

## 形状

- `len()`：返回对象的长度
  - `len([[1,2,3],[3,4,5]])`，返回值为2
- `count()`：计算包含对象个数 
  - `[1,1,1,2].count(1)`，返回值为3 
- numpy
  - `size()`：计算数组和矩阵所有数据的**个数** 
  - `shape ()`:得到矩阵**每维的大小** 

## range()

- 创建一个整数列表，一般用在 for 循环中
- 返回range对象，如想返回一个list，前面加上list转换

- range(start,stop,step)
  - start: 计数从 start 开始。默认是从 0 开始。例如range（5）等价于range（0， 5）
  - stop: 计数到 stop 结束，但不包括 stop。例如：range（0， 5） 是[0, 1, 2, 3, 4]没有5
  - step：步长，默认为1，只能是整数。例如：range（0， 5） 等价于 range(0, 5, 1)

```python
x = 0
for i in range(101):
	x = x + i
```

### arange

- numpy模块中的函数
- 返回**array类型**对象
- np.arange()中的步长**可以为小数**

```python
np.arange(2, 10, 0.5)
array([ 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5])
```



## lambda

​	目的：需要一个函数但又不想命名一个函数（匿名函数），冒号右面是返回值，例如

```python
# 基本语法
lambda [arg1 [,arg2,.....argn]]:expression

# map() 将一个函数映射到一个可枚举类型上面
# map(f, a)，将函数f依次套用在a的每一个元素上
map(lambda x: x+1 , [1,2,3])
# [2,3,4]
```

​	只能由一条表达式组成，如果只打算给其中一部分参数设定默认值，那么应当将其放在靠后的位置（和定义函数时一样，避免歧义），否则会报错。

## id( )

​	用于获取对象内存地址。

## format()

- `fstring`替换 

格式化字符串，基本语法是通过 {} 和 : 来代替以前的 %

```python
"{1} {0} {1}".format("hello","world")
# world hello world
# {} 指定替代位置
print("网：{name}, 地 {url}".format(name="菜", url="www.runoob.com"))
# 可以用变量名
print("{:.2f}".format(3.1415926));
# 3.14
# ：辅助控制格式
```

## 枚举(enumerate)

- 参考资料
  - [基本用法][https://zhuanlan.zhihu.com/p/61077440]
- 允许遍历数据并自动计数

+ `enumerate(sequence, [start=0])`

```python
# 可选参数 '1'指定从1开始枚举
my_list = ['apple', 'banana', 'grapes', 'pear']
for c, value in enumerate(my_list, 1):
    print(c, value)
"""
(1, 'apple')
(2, 'banana')
(3, 'grapes')
(4, 'pear')
"""
```

## zip()

- 用于将可迭代对象作为参数，将对象中对应的元素打包成元祖，返回由元祖组成的对象

  - `zip([iterable, ...])`
    - `iterable`一个或多个迭代器

- 输出

  - 返回一个`zip`对象

  - 可以用`list()`转换输出列表

  - 利用`*`号操作符，可以将元祖解压为列表

  - ```python
    a = [1, 2, 3]
    c = [4, 5, 6, 7, 8]
    zipped = zip(a, c)
    [(1, 4), (2, 5), (3, 6)]
    # 元素个数和最短列表一致
    ```

- 样例

  - ```python
    X = [1, 2, 3, 4, 5, 6]
    y = [0, 1, 0, 0, 1, 1]
    zipped_data = list(zip(x, y))
    new_listed_data = list(map(list, zip(*zipped_data)))
    # zip(*)反向解压，map()逐项转换类型，list()做最后转换
    # new_listed_data [[1, 2, 3, 4, 5, 6], [0, 1, 1, 0, 0, 1]]
    # zipped_data [(1, 0), (2, 1), (3, 1), (4, 0), (5, 0), (6, 1)]
    ```

  - [教程](https://static.kancloud.cn/smilesb101/python3_x/296157)

## join()

- 以**指定的字符**连接序列中的元素生成一个新的字符串

- `str.join(sequence)`

  - `sequence ` 必须是字符串

- ```python
  s1 = "-"
  s2 = ""
  seq = ("r", "u", "n", "o", "o", "b") # 字符串序列
  print (s1.join( seq ))
  print (s2.join( seq ))
  >>>r-u-n-o-o-b
  >>>runoob
  ```

- 广义上可迭代对象的例子包括

  - 所有序列类型（例如 `list`、`str` 和 `tuple`）
  - 以及某些非序列类型例如 `dict`、文件对象 以及
  - 定义了```__iter__() ```方法**或**是实现了 Sequence 语义的  `__getitem__() `方法的任意自定义类对象



## strip()

- 移除字符串头尾指定的字符(默认为空格)或字符序列

  - 只删开头不删中间

- 语法

  - ```python
    str.strip([chars])
    ```

- 返回移除字符串头尾指定的字符序列生成的新字符串



## 计数器(counter)

- Counter是一个容器对象,主要的作用是用来统计散列对象,可以使用三种方式来初始化

  1. 参数里面参数可迭代对象 `Counter("success")`
  2. 传入关键字参数`Counter((s=3,c=2,e=1,u=1))`
  3. 传入字典 `Counter({"s":3,"c"=2,"e"=1,"u"=1})`

- ```python
  a = [1,4,2,3,2,3,4,2]
  print Counter(a).most_commo（4）
  Out[10]: [(2, 3), (4, 2), (3, 2), (1, 1)]
      
  c = Counter(['eggs', 'ham'])
  c['bacon']                              # 不存在就返回0
  Out[10]:0
  # 获得所有元素
  c = Counter(a=4, b=2, c=0, d=-2)
  list(c.elements())
  Out[10]:['a', 'a', 'a', 'a', 'b', 'b']
  
  c & d                       # 求最小
  #Counter({'a': 1, 'b': 1})
  c | d                       # 求最大
  #Counter({'a': 3, 'b': 2})
  ```

- 例子：读文件统计词频并按照出现次数排序，文件是以空格隔开的单词的诸多句子：

  ```python
  from collections import Counter
  lines = open("./data/input.txt","r").read().splitlines()
  lines = [lines[i].split(" ") for i in range(len(lines))]
  words = []
  for line in lines:
      words.extend(line)
  result = Counter(words)
  print (result.most_common(10))
  ```

## 迭代器和生成器

迭代是**访问集合元素**的方式；迭代器是可以记住**遍历的位置**的对象；迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束；**字符串，列表或元组对象都可用于创建迭代器。**

​	迭代器有两个基本方法：

​		iter()

​		next()

```python
list = [1,2,3,4]
it = iter(list)
print(next(it))  # 1
print(next(it))  # 2
```

把一个类作为一个迭代器使用:

```python
class MyNumbers:
    def __iter__(self):
		self.a = 1
        return self
   	def __next__(self):
        if self.a <= 20:
            x = self.a
        	self.a += 1
        	return x
        else:
            raise StopIteration
myclass = MyNumbers()
myiter = iter(myclass)

for x in myiter:
    print(x)
```

生成器(generator)是使用`yiled()`函数一个返回迭代器的函数，只能用于迭代操作。

```python
import sys
def fib(n):
    a,b,counter = 0,1,0
    while(True):
        if(counter > n):
            return
        yield a
        # 遇到yiled()函数暂停并保存运行信息，返回yield值，下一次执行next()时从当前位置开始
        a,b = b,a+b
        counter += 1
f = fib(10)
while(True):
    try:
        print(next(f),end=" ")
    except StopIteration:
        sys.exit()
```

## 数学函数

abs(x) 绝对值

exp(x) e的x次幂

fabs(x) 绝对值

log(a,b) 以b为底对数

log10(x) 以10为底的对数

max()\min() 返回最值

pow(x,y) x的y次幂

round(x,[n]) 

​	浮点数x的四舍五入值，n表示舍入到小数点后的位数

sqrt(x) x的平方根

ceil() 向上取整

## 随机数函数

choice(sew)

```python
random.choice(range(10))
# 0到9中随机挑选一个整数
```

random()随机生成[0,1)范围内实数

randrange(start, stop,step)

```python
random.randrange(1,100,2)
# start -- 开始值，包含
# stop -- 结束值，不包含
# step -- 递增基数
```

## 返回最值

### 返回最大值索引

- Returns the **indices** of the maximum values along an axis.

- ```python
  numpy. argmax(a, axis=None, out=None)
  ```

### 返回最大值

- Return the maximum of an array or maximum along an axis.

- ```python
  a = np.arange(4).reshape((2,2))
  a
  array([[0, 1],
         [2, 3]])
  np.amax(a)           # Maximum of the flattened array
  3
  np.amax(a, axis=0)   # Maxima along the first axis
  array([2, 3])
  np.amax(a, axis=1)   # Maxima along the second axis
  array([1, 3]
  ```

## 数学常量

pi	π

e	自然常数

## assert 断言

- 用于判断一个表达式，在**表达式条件为 false** 的时候触发异常

- ```python
  assert expression
  ```

## @ 装饰器

- 函数，提供调用
- [阅读材料1](https://blog.csdn.net/Sean_ran/article/details/52097997?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-6.channel_param&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-6.channel_param)

# 常用库

## numpy

numpy是Python中科学计算的核心库，提供高位矩阵计算工具

## Matplotlib

画图工具，2D，3D

```python
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,8,0.1)
y = np.sin(x)
plt.plot(x,y)
plt.show()
```

 ![python_10](F:\机器学习笔记\工具笔记\picformd\python_10.PNG)

## Pillow

Python官方图像处理库，使用 pip install Pillow安装

```python
from PTL import Image
img = Image.open('./picformd/python_10.PNG')
img

```

## Pathlib

- [博客][https://www.dongwm.com/post/use-pathlib/]

- [pathlib路径库使用详解][https://xin053.github.io/2016/07/03/pathlib%E8%B7%AF%E5%BE%84%E5%BA%93%E4%BD%BF%E7%94%A8%E8%AF%A6%E8%A7%A3/]

- [官网][https://docs.python.org/zh-cn/3.6/library/pathlib.html]

- import

  - `from pathlib import Path`

- ```python
  from pathlib import Path
  p = Path()
  # WindowsPath('.')
  p.resolve()                     # 文档显示是absolute path, 这里感觉只能用在获取当前绝对路径上
  # WindowsPath('C:/Users/Cabby')
  ```


- 和`os and os.path`对照表

  
  - [ospath][https://docs.python.org/zh-cn/3.6/library/os.path.html#module-os.path]
  
  - | os and os.path           | pathlib                                 |
    | :----------------------- | :-------------------------------------- |
    | `os.path.abspath`        | `Path.resolve`                          |
    | `os.chmod`               | `Path.chmod`                            |
    | `os.mkdir`               | `Path.mkdir`                            |
    | `os.rename`              | `Path.rename`                           |
    | `os.replace`             | `Path.replace`                          |
    | `os.rmdir`               | `Path.rmdir`                            |
    | `os.remove`, `os.unlink` | `Path.unlink`                           |
    | `os.getcwd`              | `Path.cwd`                              |
    | `os.path.exists`         | `Path.exists`                           |
    | `os.path.expanduser`     | `Path.expanduser` and `Path.home`       |
    | `os.path.isdir`          | `Path.is_dir`                           |
    | `os.path.isfile`         | `Path.is_file`                          |
    | `os.path.islink`         | `Path.is_symlink`                       |
    | `os.stat`                | `Path.stat`, `Path.owner`, `Path.group` |
    | `os.path.isabs`          | `PurePath.is_absolute`                  |
    | `os.path.join`           | `PurePath.joinpath`                     |
    | `os.path.basename`       | `PurePath.name`                         |
    | `os.path.dirname`        | `PurePath.parent`                       |
    | `os.path.samefile`       | `Path.samefile`                         |
    | `os.path.splitext`       | `PurePath.suffix`                       |

# PyCharm debug 问题

1、必须设置断点

2、断点标记变为蓝色意味着已经达到了断点，此时**尚未**执行突出显示的代码行。