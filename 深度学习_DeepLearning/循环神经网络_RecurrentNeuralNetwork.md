
# 循环神经网络_RecurrentNeuralNetwork

### 结构
给定输入序列 $x_1, x_2, ..., x_T$，模型输出序列 $y_1, y_2, ..., y_T$

$$
\begin{aligned}
h_t &= \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h) \quad &\text{(隐藏状态更新)} \\
y_t &= W_{hy} h_t + b_y \quad &\text{(输出)} \\
\end{aligned}
$$


- 输入维度：$x_t \in \mathbb{R}^{d_x}$
- 隐藏状态维度：$h_t \in \mathbb{R}^{d_h}$
- 输出维度：$y_t \in \mathbb{R}^{d_y}$

则：

| 参数         | 维度                     |
|--------------|--------------------------|
| $W_{xh}$ | $d_h \times d_x$     |
| $W_{hh}$ | $d_h \times d_h$     |
| $b_h$    | $d_h$                |
| $W_{hy}$ | $d_y \times d_h$     |
| $b_y$    | $d_y$                |

---

### 前向传播过程说明

给定一个长度为 $T$ 的输入序列：

1. 初始化隐藏状态 $h_0 = \mathbf{0}$  
2. 对每个时间步 $t = 1, 2, ..., T$：

```python
for t in range(T):
    h[t] = tanh(Wxh @ x[t] + Whh @ h[t-1] + bh)
    y[t] = Why @ h[t] + by
```

---

### 反向传播 Through Time（BPTT）

核心思想是：**链式法则地传播误差回每个时间步**。


假设我们使用 MSE（均方误差）损失：

$$
L = \frac{1}{2} \sum_t \| y_t - \hat{y}_t \|^2
$$


我们定义隐藏状态的误差为：

$$
\delta^h_t = \frac{\partial L}{\partial h_t} \in \mathbb{R}^{d_h}
$$

这个误差来源于两部分：

1. 当前时间步的输出层反传：  
   $$
   \delta^h_t += W_{hy}^T \cdot \delta^y_t
   $$
   其中 $\delta^y_t = y_t - \hat{y}_t$

2. 下一时刻隐藏层的误差项：
   $$
   \delta^h_t += W_{hh}^T \cdot \delta^h_{t+1} \cdot (1 - h_{t+1}^2)
   $$

最终：

$$
\delta^h_t = (W_{hy}^T \cdot \delta^y_t + W_{hh}^T \cdot \delta^h_{t+1} \cdot (1 - h_{t+1}^2)) \odot (1 - h_t^2)
$$

---

### 梯度计算公式

对每个时间步 $t$：

$$
\begin{aligned}
\frac{\partial L}{\partial W_{hy}} &= \sum_t \delta^y_t h_t^T \\
\frac{\partial L}{\partial b_y} &= \sum_t \delta^y_t \\
\frac{\partial L}{\partial W_{xh}} &= \sum_t \delta^h_t x_t^T \\
\frac{\partial L}{\partial W_{hh}} &= \sum_t \delta^h_t h_{t-1}^T \\
\frac{\partial L}{\partial b_h} &= \sum_t \delta^h_t \\
\end{aligned}
$$

---

###  梯度下降（SGD）

每次迭代（或 mini-batch）后进行参数更新：

$$
\theta \gets \theta - \eta \cdot \frac{\partial L}{\partial \theta}
$$

即对所有参数 $W_{xh}, W_{hh}, W_{hy}, b_h, b_y$：

```python
Wxh -= lr * dWxh
Whh -= lr * dWhh
Why -= lr * dWhy
bh  -= lr * dbh
by  -= lr * dby
```


# 长短期记忆网络LongShort-TermMemory

[推荐阅读](https://zh.d2l.ai/chapter_recurrent-modern/lstm.html)

LSTM 的关键思想是引入了 **记忆单元（Cell State）** 和 **门控机制**，让网络能够学习长期依赖关系。

## LSTM 的结构组件

LSTM 的基本结构包含四个部分：

- **输入门**（input gate）$i_t$
- **遗忘门**（forget gate）$f_t$
- **输出门**（output gate）$o_t$
- **候选记忆单元**（cell candidate）$\tilde{c}_t$
- **记忆单元状态**（cell state）$c_t$
- **隐藏状态**（hidden state）$h_t$

---

## 三、LSTM 关键公式详解

设当前时刻为 $t$，输入为 $x_t$，上一个隐藏状态为 $h_{t-1}$，上一个记忆状态为 $c_{t-1}$。

我们逐步构造 LSTM 的计算流程：

---

### 1️⃣ 遗忘门（Forget Gate）

决定前一时刻的记忆有多少被保留：

$$
f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f)
$$

- $W_f$ 是输入的权重，$U_f$ 是隐藏状态的权重；
- $\sigma$ 是 sigmoid 激活函数，输出在 $[0, 1]$；
- $f_t$ 越接近 1，说明信息被保留；越接近 0，说明信息被遗忘。

---

### 2️⃣ 输入门（Input Gate）

决定当前输入中有多少被写入记忆：

$$
i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i)
$$

---

### 3️⃣ 候选值（Candidate State）

生成一个当前输入的候选记忆向量：

$$
\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c)
$$

- $\tilde{c}_t$ 是当前时刻可以被写入的新内容；
- 输入门 $i_t$ 会控制其中多少被写入。

---

### 4️⃣ 记忆单元状态更新（Cell State）

将旧的记忆 $c_{t-1}$ 与新的候选值融合形成新的记忆 $c_t$：

$$
c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t
$$

- $\odot$ 表示按元素乘（Hadamard product）；
- 旧记忆按遗忘门保留，新信息按输入门写入。

---

### 5️⃣ 输出门（Output Gate）

决定要输出多少当前隐藏状态 $h_t$：

$$
o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o)
$$

---

### 6️⃣ 最终隐藏状态 $h_t$

当前的隐藏状态是经过激活的记忆状态，与输出门相乘：

$$
h_t = o_t \odot \tanh(c_t)
$$
