
## 一、神经网络的基本结构

### 1.1 感知机模型（Perceptron）

最基本的神经元模型为[感知机](感知机_Perceptron)，计算过程如下：

$$
a = \boldsymbol{w}^T \boldsymbol{x} + b
$$$$
z = \phi(a)
$$
### 1.2 多层感知机（MLP）

神经网络由多个**神经元组成的层（Layer）** 构成：

- **输入层**：接收原始特征
- **隐藏层**：进行特征抽象与非线性变换
- **输出层**：输出预测结果（分类或回归）


## 二、激活函数（Activation Functions）

[激活函数](MLFoundations#4.激活函数_ActivationFunctions)引入非线性，使网络具备拟合复杂函数的能力。

## 三、前向传播ForwardPropagation

在训练过程中，样本输入网络后，按层依次计算输出：

假设神经网络共有$L$ 层，对于第 $l$ 层：
$$
a^{(l)} = W^{(l)} z^{(l-1)} + b^{(l)}
$$
$$
z^{(l)} = \phi(a^{(l)})
$$

或者：
$$
a^{(l)} = W^{(l)} z^{(l-1)}
$$
$$
z^{(l)} = [1,\phi(a^{(l)})]^T
$$最终输出为预测值 $\hat{y} = z^{(L)} = h_{W^{(1)}\dots W^{(L)}}(x)$  其中$h$ 代表hidden state

## 四、损失函数（Loss Function）
[损失函数](MLFoundations#1.损失函数_LossFunctions)
## 五、反向传播（Backpropagation）

我们假设第 $l$ 层的计算结构如下：

1. 接收输入 $z^{[l-1]}$
2. 使用线性变换得出中间量 $a^{[l]}$：

   $$
   a^{[l]} = W^{[l]} z^{[l-1]} + b^{[l]}
   $$

3. 激活函数作用在 $a^{[l]}$ 上，定义输出 $z^{[l]}$ 为：

   $$
   z^{[l]} = \phi(a^{[l]})
   $$



设最终输出层为 $L$，网络预测输出为：

$$
\hat{y} = z^{[L]}
$$

设损失函数为 $\mathcal{L}(\hat{y}, y)$。


我们希望计算每层参数的梯度：

- $\frac{\partial \mathcal{L}}{\partial W^{[l]}}$
- $\frac{\partial \mathcal{L}}{\partial b^{[l]}}$
- 以及中间变量 $\frac{\partial \mathcal{L}}{\partial z^{[l]}}$

由链式法则，我们有：

$$
\frac{\partial \mathcal{L}}{\partial W^{[l]}} = \frac{\partial \mathcal{L}}{\partial z^{[l]}} \cdot \frac{\partial z^{[l]}}{\partial a^{[l]}} \cdot \frac{\partial a^{[l]}}{\partial W^{[l]}}
$$

同理：

$$
\frac{\partial \mathcal{L}}{\partial z^{[l-1]}} = \frac{\partial \mathcal{L}}{\partial z^{[l]}} \cdot \frac{\partial z^{[l]}}{\partial a^{[l]}} \cdot \frac{\partial a^{[l]}}{\partial z^{[l-1]}}
$$


对于输出层 $L$：

$$
\delta^{[L]} = \frac{\partial \mathcal{L}}{\partial z^{[L]}}\frac{\partial z^{[L]}}{\partial a^{[L]}} = \frac{\partial \mathcal{L}}{\partial \hat{y}}\frac{\partial z^{[L]}}{\partial a^{[L]}} \quad \text{(依据损失函数定义)}
$$


我们定义第 $l$ 层的“误差项”为：

$$
\delta^{[l]} = \frac{\partial \mathcal{L}}{\partial z^{[l]}}\frac{\partial z^{[l]}}{\partial a^{[l]}}
$$

然后向前传播误差（即向后传播梯度）：

$$
\delta^{[l-1]} = \left(W^{[l]}\right)^T \cdot \left(\delta^{[l]} \circ \phi'(a^{[l-1]}) \right)
$$
解释：

- $\delta^{[l]}$ 是来自上一层的梯度；
- $\phi'(a^{[l]})$ 是当前层的激活函数导数；
- $\circ$ 是元素乘（Hadamard 积）。

---

### Step 3：计算参数梯度

根据链式法则继续展开：

- 权重的梯度：

  $$
  \frac{\partial \mathcal{L}}{\partial W^{[l]}} = \left( \delta^{[l]} \circ \phi'(a^{[l]}) \right) \cdot \left( z^{[l-1]} \right)^T
  $$

- 偏置的梯度：

  $$
  \frac{\partial \mathcal{L}}{\partial b^{[l]}} = \delta^{[l]} \circ \phi'(a^{[l]})
  $$



## 五、参数更新公式（以学习率 $\eta$ 为例）

最终使用梯度下降法更新参数：

$$
W^{[l]} := W^{[l]} - \eta \cdot \frac{\partial \mathcal{L}}{\partial W^{[l]}}
$$

$$
b^{[l]} := b^{[l]} - \eta \cdot \frac{\partial \mathcal{L}}{\partial b^{[l]}}
$$

对于本例中每一层，执行上述更新即可。

---

## 六、小结

| 步骤 | 内容 |
|------|------|
| 前向传播 | 计算每层输出 $z^{[l]}, a^{[l]}$ |
| 反向传播 | 自输出层向前逐层计算 $\delta^{[l]}$ |
| 梯度计算 | 利用 $\delta$ 计算 $\frac{\partial \mathcal{L}}{\partial W^{[l]}}$ |
| 参数更新 | 按学习率 $\eta$ 更新权重和偏置 |

---


## 七、训练过程概览

1. 初始化权重参数
2. 进行前向传播，计算预测值
3. 计算损失
4. 反向传播，计算梯度
5. 更新参数
6. 重复以上步骤，直到收敛或满足停止条件

## 八、过拟合与正则化

神经网络容易过拟合，常见应对方式：

- L2/L1 [正则化](正则化_Regularization)（限制参数大小）
- Dropout（随机丢弃部分神经元）
- 数据增强（扩充训练样本）
- 提前停止（EarlyStopping）


## 九、深度神经网络（DNN）

当隐藏层数增加时，神经网络演变为深度神经网络，具备更强的特征表达能力，但也更难训练。

现代深度学习框架（如 TensorFlow、PyTorch）支持构建任意复杂结构的 DNN。

