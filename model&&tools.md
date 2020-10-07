- pytorch

# 线性转换

- $y=xA^T + b$

- ```python
  CLASS torch.nn.Linear(in_features: int, out_features: int, bias: bool = True)
  ```

  - **in_features** – size of each input sample
  - **out_features** – size of each output sample
  - **bias** – If set to `False`, the layer will not learn an additive bias. Default: `True`

- shape

  - $Input: (N, *, H_{in}) $ 
    - $H_{in}$  in_features
  - $Output: (N, *, H_{out})$
    - $H_{out}$ out_features
    - 前两维和 Input 前两维相同

- 参数

  - `Linear.weight` （out_features,in_features）
  - `Linear.bias` (out_features)
    - If `bias` is `True`, the values are initialized from $\mathcal{U}(-\sqrt{k}, \sqrt{k})$ where $k = \frac{1}{\text{in_features}}$

# RNN

- all RNN modules accept packed sequences as inputs

## 工具

### 处理变长序列

- ```python
  torch.nn.utils.rnn.pad_sequence(sequences, batch_first=False, padding_value=0.0)
  ```

  - 把长度小于最大长度的 sequences 用 `padding_value` 填充，并且把 list 中所有的元素拼成一个 tensor
  
  - ```python
    >>> from torch.nn.utils.rnn import pad_sequence
    >>> a = torch.ones(25, 300)
    >>> b = torch.ones(22, 300)
    >>> c = torch.ones(15, 300)
    >>> pad_sequence([a, b, c]).size()
    torch.Size([25, 3, 300])
    ```
  
    - a list of sequence with size `L x *` ---> `T x B x *` 
      - **L** :  sentence 长度 **T**：最长sentence长度 **B**：batch_size
    -  True : output  `B x T x *` , False :  `T x B x *`
  
- ```python
  torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True)
  ```

  - Packs 一个变长填充后 sequences 的 Tensor
    - <img src="https://img2018.cnblogs.com/blog/1404499/201810/1404499-20181022223531597-1352808756.png" alt="img" style="zoom:50%;" />
  - shape
    - input: `T x B x *` 
      - T：最长序列的长度
      - B：batch size
  - 未排序序列， 使用`enforce_sorted = False` ；如果`enforce_sorted = True` ，sequences要按长度降序排列
  - 注意
    - 通过访问`PackedSequence.data` 属性恢复Tensor
  
- ```python
  torch.nn.utils.rnn.pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None)
  ```

  - inverse operation to pack_padded_sequence

#### 阅读材料

- [pad_sequence、pack_padded_sequence、pad_packed_sequence](https://zhuanlan.zhihu.com/p/59772104)


#### 示例代码

```python
import torch.nn.utils.rnn as rnn_utils
train_x = [torch.Tensor([1, 1, 1, 1, 1, 1, 1]),
           torch.Tensor([2, 2, 2, 2, 2, 2]),
           torch.Tensor([3, 3, 3, 3, 3]),
           torch.Tensor([4, 4, 4, 4]),
           torch.Tensor([5, 5, 5]),
           torch.Tensor([6, 6]),
           torch.Tensor([7])
           ]

class MyData(data.Dataset):
    def __init__(self, data_seq):
        self.data_seq = data_seq

    def __len__(self):
        return len(self.data_seq)

    def __getitem__(self, idx):
        return self.data_seq[idx]
   
def collate_fn(data):
    data.sort(key=lambda x: len(x), reverse=True)
    data_length = [len(sq) for sq in data]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0)
    return data.unsqueeze(-1), data_length

if __name__=='__main__':
    data = MyData(train_x)
    data_loader = DataLoader(data, batch_size=3, shuffle=True, collate_fn=collate_fn)
    batch_x, batch_x_len = iter(data_loader).next()
    batch_x_pack = rnn_utils.pack_padded_sequence(batch_x, batch_x_len, batch_first=True)
    net = nn.LSTM(1, 10, 2, batch_first=True)
    h0 = torch.rand(2, 3, 10)
    c0 = torch.rand(2, 3, 10)
    out, (h1, c1) = net(batch_x_pack, (h0,c0))
    out_pad, out_len = rnn_utils.pad_packed_sequence(out, batch_first=True)
    print('END')
```

##LSTM

```python
class torch.nn.LSTM(*args, **kwargs)
```

- Input : input, (h_0, c_0)
  - **input**：(seq_len, batch, input_size)
  - **h_0**： (num_layers * num_directions, batch, hidden_size) 
    - **hidden state**
    - num_layer：几层
    - num_directions：双向还是单向
  - **c_0**：(num_layers * num_directions, batch, hidden_size)
    - **cell state**
- Output: output, (h_n, c_n)
  - **output**：(seq_len, batch, num_directions \* hidden_size)
  - **h_n**： (num_layers * num_directions, batch, hidden_size)
  - **c_n**：(num_layers * num_directions, batch, hidden_size)

