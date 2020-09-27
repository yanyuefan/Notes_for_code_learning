# 基础

- 导入

`import pandas as pd`

- `csv\ tsv`
  - csv分隔`,`
  - `tsv`分隔`\t`
  
- 支持数据类型

  - ```python
    1. float 
    2. int 
    3. bool 
    4. datetime64[ns] 
    5. datetime64[ns, tz] 
    6. timedelta[ns] 
    7. category 
    8. object 
    默认的数据类型是int64,float64.
    ```

    

##读取文件

- 具体参数设置[随用随搜](https://www.jianshu.com/p/5f058d535e6d)

- ```python
  csv_data = pd.read_csv('./data/Students_red.csv',index_col='ID')
  print(csv_data)
  ```

- ```python
  tsv_data = pd.read_csv('./data/Students_red.tsv',sep='\t',index_col='ID')
  print(tsv_data)
  ```

- ```python
  txt_data = pd.read_csv('./data/Students_red.txt',sep='|',index_col='ID')
  print(txt_data)
  ```

- ```python
  video_data = pd.read_excel('./data/Videos.xlsx',index_col='Month')
  print(video_data)
  ```

- 数据表转置

  - ```python
    # dataframe 转置
    video_new_dataa = video_data.T
    print(video_new_dataa)
    ```

- ```python
  data = pd.read_csv( my_file.csv , sep= ; , encoding= latin-1 , nrows=1000, skiprows=[2,5])
  # sep 代表的是分隔符。如果你在使用法语数据，excel 中 csv 分隔符是「;」，因此你需要显式地指定它。编码设置为 latin-1 来读取法语字符。nrows=1000 表示读取前 1000 行数据。skiprows=[2,5] 表示你在读取文件的时候会移除第 2 行和第 5 行。
  ```

## 写数据

- ```python
  DataFrame.to_csv(path_or_buf=None, sep=', ', na_rep='', float_format=None, columns=None, header=True, index=True,
                   index_label=None, mode='w', encoding=None, compression=None, quoting=None, quotechar='"',
                   line_terminator='\n', chunksize=None, tupleize_cols=None, date_format=None, doublequote=True,
                   escapechar=None, decimal='.')
  """
  参数：
  path_or_buf : 文件路径，如果没有指定则将会直接返回字符串的 json
  sep : 输出文件的字段分隔符，默认为 “,”
  na_rep : 用于替换空数据的字符串，默认为''
  float_format : 设置浮点数的格式（几位小数点）
  columns : 要写的列
  header : 是否保存列名，默认为 True ，保存
  index : 是否保存索引，默认为 True ，保存
  index_label : 索引的列标签名
  """
  ```

- ```python
  data.to_csv( my_new_file.csv , index=None)
  '''
  index=None 表示将会以数据本来的样子写入。如果没有写 index=None，你会多出一个第一列，内容是 1，2，3，...，一直到最后一行。
  '''
  ```

- 写入到excel文件

  - `DaraFrame.to_excel()`

- ```python
  DataFrame.to_excel(excel_writer, sheet_name='Sheet1', na_rep='', float_format=None, columns=None, 
                     header=True, index=True, index_label=None, startrow=0, startcol=0, engine=None,
                     merge_cells=True, encoding=None, inf_rep='inf', verbose=True, freeze_panes=None)
  ```

  

## 检查数据

- ```python
  data.describe()  # 计算基本统计数据
  data.columns  # 打印全部列的名称
  data[["c_name"]]  # 读出的数据是DataFrame类型
  data.index  # 数据集中的索引
  """RangeIndex(start=0, stop=4622, step=1)"""
  data.info()  
  # 数据整体情况 RangeIndex；Columns；dtypes；memory usage
  data.dtypes  # 查看数据类型
  ```

- ```python
  data.head(3)  # 打印前三行
  data.tail(3)  # 打印后三行
  data.loc[8]  # 打印第8行
  data.loc[8, c_name]  # 打印第8行名为[c_name]的列
  data.loc[range(4,6)]  # 第四到第六行（左闭右开）的数据子集 [4,6)
  ```

- ```python
  """逻辑运算"""
  data[(data[ column_1 ]== french ) & (data[ year_born ]==1990)\
       & ~(data[ city ]== London )]
  # 通过逻辑运算来取数据子集。要使用 & (AND)、 ~ (NOT) 和 | (OR)，必须在逻辑运算前后加上「and」。
  ```

## 更新数据

- ```python
  data.loc[8, c_name] = "e"  # 将c_name下的第8行元素替换为"e"
  data.loc[data[ c_name ]== french , c_name_1 ] = French
  # 将c_name列为"french"的行，c_name_1的元素替换为"French",不指定c_name_1则整行元素替换为"French"
  df.iat[0,1] = 1
  df[df > 0] = 1
  
  df['A'] = 0  # 只能修改一行/多行的值，一列或者多列的值
  """按照index添加新列"""
  s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
  data['F'] = s1
  ```

- ```python
  """全数据操作"""
  data[c_name].map(len)  # 
  ```

- ```python
  data.columns.tolist()  #转换为list
  ```

- ```python
  """删除 drop"""
  df.drop(['index1','index2'])  # 删除多个index
  df.drop(['column1','column2']，axis=1)  # 删除column axis=1 就是列的值
  ```

  

## 统计出现的次数

- ```python
  data[c_name].value_counts
  ```

- ```python
  data.shape[0]  #行数 0维特征数
  data.shape[1]  #列数 1维特征数
  ```

  

## 创建对象

- 两个数据结构
  - series
  - `DataFrames`

### series

-  a list of values 值序列，只有一列
- create a default integer index

```python
s = pd.Series([1,3,5,np.nan,6,8])  # list

0    1.0
1    3.0
2    5.0
3    NaN
4    6.0
5    8.0
dtype: float64
```

### dataframe

- 有多个列的数据表，每列有一个label，也有索引

```python
dates = pd.date_range('20130101', periods=6)
df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))

                 A          B                      C                D
2013-01-01  -0.791274   -1.036452   -0.442802   -0.348499
2013-01-02  0.301447    -1.294061   0.527779    -0.989473
2013-01-03  1.194980    1.032717    -0.747851   0.483149
2013-01-04  1.080179    0.760932    -0.660007   0.901296
2013-01-05  0.750439    -1.494272   1.108947    0.057424
2013-01-06  0.431982    -0.829019   1.166722    0.216905
```

#### 通过字典`dict`转变为表格

- ```python
  web_stats = {'Day':[1,2,3,4,5,6],
               'Visitors':[43,34,65,56,29,76],
               'Bounce Rate':[65,67,78,65,45,52]}
  
  df = pd.DataFrame(web_stats)
  ####
     Day  Visitors  Bounce Rate
  0    1        43           65
  1    2        34           67
  2    3        65           78
  3    4        56           65
  4    5        29           45
  ```

##### 通过**字典和zip**创建表格

- ```python
  cities = ['austin','dallas','austin','dallas']
  visitors =[139,237,326,456]
  list_labels = ['city','visitors']
  list_cols = [cities,visitors]
  
  zipped =list(zip(list_labels,list_cols))
  data = dict(zipped)
  
  users = pd.DataFrame(data)
  users
  ####
      city    visitors
  0   austin  139
  1   dallas  237
  2   austin  326 
  3   dallas  456
  ```

##### 通过嵌套字典创建

- ```python
  """外层字典的键作为列，内层键作为行索引"""
  
  pop = {'Nevada':{2001:2.4,2002:2.9},'Ohio':{2000:1.5,2001:1.7,2002:3.6}}
  df= pd.DataFrame(pop)
  ####
       Nevada  Ohio
  2000    NaN 1.5
  2001    2.4 1.7
  2002    2.9 3.6
  ```

#### 运算

```python
data.add() # 加法
data.sub() # 减法
data.div() # 除法
data.mul() # 乘法
```



## 读取行/列：切片

- 直接`[ ]`

  - `data['A']`
  - `data[0:3]`

- 访问单个元素`data.at()`

  - ```python
    data.at[1, 'city']  
    # rows==1&&col_label == 'city'
    ```

  - rows: index；columns: 列名，不能是position

- 访问单个元素`data.iat()`

  - ```python
    data.iat[1, 2]  # 1行2列
    ```

- `data.loc`

  - loc[ ]括号里先行后列，`,`分割，行、列分别是**行标签和列标签**
  - 多行/列 用`:`取值
  - loc在对`DataFrame`进行重新赋值操作时会避免**chained indexing**问题

- `data.iloc()`

  - `.iloc[]`与loc一样，**中括号**里面也是先行后列，行列标签用`,`分割，与loc不同的之处是，`.iloc` 是根据**行数与列数**来索引的
  - 多行多列用`:` 
  - 关键字用数字表示，注意python从0开始！！！

- 对bool值

  - ```python
    df[df.A>0]
    """
    A   B
    2013-01-01  0.827999    1.153047
    2013-01-04  0.426128    -1.253330
    """
    ```

## index

![img](https://upload-images.jianshu.io/upload_images/10450029-fa610316937e7a08.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)



- `reindex`

  - 创建一个适应新索引的新对象

  - ```python
    """series"""
    obj = pd.Series([4.5,7.2,-5.3,3.6],index=['d','b','a','c'])
    obj
    d    4.5
    b    7.2
    a   -5.3
    c    3.6
    dtype: float64
    
    obj2 = obj.reindex(['a','b','c','d','e'], fill_value = 0)
    obj2
    a   -5.3
    b    7.2
    c    3.6
    d    4.5
    e    0.0
    dtype: float64
    ```
```
    
  - ![img](https://upload-images.jianshu.io/upload_images/10450029-5eab103a10988233.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)
  
  - dataframe
  
  - ```python
    frame = pd.DataFrame(np.arange(9).reshape((3,3)),index = ['a','c','d'],columns = ['Ohio','Texas','California'])
    ####
        Ohio    Texas   California
    a   0   1   2
    c   3   4   5
    d   6   7   8
    
    frame2 = frame.reindex(['a','b','c','d']) #没有的就用NaN代替
    ####
        Ohio    Texas   California
    a   0.0 1.0 2.0
    b   NaN NaN NaN
    c   3.0 4.0 5.0
    d   6.0 7.0 8.0
    
    """使用关键字重新索引列"""
    states = ['Texas','Utah','California'] 
    frame.reindex(columns = states)
    ####
      Texas Utah    California
    a   1   NaN 2
    c   4   NaN 5
    d   7   NaN 8
```

    - 属性![img](https://upload-images.jianshu.io/upload_images/10450029-c937ae7f16b9e4ec.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200/format/webp)

- 常见函数

  - apply( )

    - apply函数可以对`DataFrame`对象进行操作，既可以作用于一行或者一列的元素，也可以作用于单个元素。

    - ```python
      df.apply(func, axis=0, broadcast=False, raw=False, reduce=None, args=(), **kwds)
      ```

    - 行 axis = 1

    - 列 axis = 0

    - func 是要在之前def 好的 或者其他已有的func

  - sample( )

    - ```python
      DataFrame.sample(n=None, frac=None, replace=False, weights=None, random_state=None, axis=None)
      
      """属性"""
      n	        是要抽取的行数。（例如n=20000时，抽取其中的2W行）
      frac		是抽取的比列。（有一些时候，我们并对具体抽取的行数不关系，我们想抽取其中的百分比，这个时候就可以选择使用frac，例如frac=0.8，就是抽取其中80%）
      replace		是否为有放回抽样，取replace=True时为有放回抽样。
      weights		这个是每个样本的权重，具体可以看官方文档说明。
      random_state
      axis		是选择抽取数据的行还是列。axis=0的时是抽取行，axis=1时是抽取列（也就是说axis=1时，在列中随机抽取n列，在axis=0时，在行中随机抽取n行）
      ```

## `concat()`

- 连接`DataFrame`对象

- 默认纵向连接

- 合并方向

  - ```python
    pd.concat([p1,p2],axis=1)
    Out[44]: 
       0  1  2    0    1    2
    0  1  1  1  3.0  3.0  3.0
    1  2  2  2  NaN  NaN  NaN
    pd.concat([p1,p2],axis=0)
    Out[45]: 
       0  1  2
    0  1  1  1
    1  2  2  2
    0  3  3  3
    ```

    

- 只想合并相同的列， 我们可以添加上join='inner'参数：

  ```python
  pd.concat([df1, df3], join='inner')
  ```

  - ![img](https://pic2.zhimg.com/80/v2-e5741cdf9ff43b357d5f211663a7cf7f_720w.jpg)
  - ![img](https://pic1.zhimg.com/80/v2-b3a5e41979b7fc58d61e2ac040d9a597_720w.jpg)

- 希望重新设置合并之后的`DataFrame`对象的index值， 可以添加`ignore_index=True`参数：

  ```text
  pd.concat([df1, df2], ignore_index=True)
  ```

- `axis=1`, 可以横向合并两个`DataFrame`对象。

  - ```python
    pd.concat([df1, df4], axis=1)
    ```

