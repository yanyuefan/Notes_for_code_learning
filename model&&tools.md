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

  - $Input: (N, *, H_{in})(N,∗,Hin) $
  - $Output: (N, *, H_{out})(N,∗,Hout) $
    - 前两维和 Input 前两维相同

- 参数

  - `Linear.weight` （out_features,in_features）
  - `Linear.bias` (out_features)
    - If `bias` is `True`, the values are initialized from $\mathcal{U}(-\sqrt{k}, \sqrt{k})U(−k,k)$ where $k = \frac{1}{\text{in_features}}$

# RNN

- all RNN modules accept packed sequences as inputs

## 工具

### 处理变长序列

- ```python
  torch.nn.utils.rnn.pad_sequence(train_x, batch_first=True)
  ```

  - 把长度小于最大长度的 sequences 用 0 填充，并且把 list 中所有的元素拼成一个 tensor

- ```python
  torch.nn.utils.rnn.pack_padded_sequence(input, lengths, batch_first=False, enforce_sorted=True)
  ```

  - 