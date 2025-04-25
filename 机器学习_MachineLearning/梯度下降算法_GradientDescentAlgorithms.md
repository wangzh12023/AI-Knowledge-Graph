
### 目录


- 批量梯度下降_BatchGradientDescent
    
- 随机梯度下降_StochasticGradientDescent
    
- 小批量梯度下降_MiniBatchGradientDescent
    
- 动量法_Momentum
    
- 自适应梯度法_AdaGrad
    
- RMSprop
    
- Adam优化器_AdamOptimizer
    
- 学习率调度_LearningRateScheduling

当然可以，以下是根据你的目录为每一项生成的 Markdown 内容（可用于笔记或知识库）：



## 梯度下降

梯度下降（Gradient Descent）是一种用于**最小化目标函数** $J(\theta)$ 的优化算法，广泛应用于机器学习模型中，如线性回归、逻辑回归、神经网络等。

基本思想是：**沿着函数梯度的负方向更新参数**，即：

$$
\theta := \theta - \eta \cdot \nabla J(\theta)
$$

其中：

- $\theta$ 是模型参数向量；
- $\eta$ 是学习率（步长）；
- $\nabla J(\theta)$ 是损失函数关于参数的梯度。

$\theta$ 是一个$n$ 维向量，所以他的$\nabla J(\theta)$ 也是一个$n$ 维向量
函数 $J(\theta)$ 对向量参数 $\theta = [\theta_1, \theta_2, \dots, \theta_M]^T$ 的梯度定义为：

$$
\nabla J(\theta) =
\begin{bmatrix}
\frac{\partial J(\theta)}{\partial \theta_1} \\\\
\frac{\partial J(\theta)}{\partial \theta_2} \\\\
\vdots \\\\
\frac{\partial J(\theta)}{\partial \theta_M}
\end{bmatrix}
$$

梯度下降更新的时候，我们让$\theta$ 的每一个维度都根据梯度方向进行一个更新

---

## 批量梯度下降（Batch Gradient Descent）

批量梯度下降是最基本的梯度下降算法。它在每次参数更新时都使用**整个训练集**计算梯度。

- **优点**：稳定，收敛方向准确  
- **缺点**：在大数据集下效率低，容易卡在局部最小值或鞍点  
- **公式**：  
- 
在每次更新中，**使用整个训练集**来计算梯度，其中 $m$ 是样本总数，$\mathcal{L}$ 是单个样本的损失：

$$
\nabla J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \nabla_\theta \mathcal{L}(\theta; x^{(i)}, y^{(i)})
$$
$$
  \theta := \theta - \eta \nabla J(\theta)
  $$


### ✅ 优点：

- 每一步都是对真实梯度的精确估计；
- 收敛过程平稳。

### ❌ 缺点：

- 每次迭代成本高，尤其是在大规模数据上；
- 无法在线学习。
---

## 随机梯度下降（Stochastic Gradient Descent, SGD）

每次迭代只用**一个样本**来更新参数。

- **优点**：计算快，有助于跳出局部最优  
- **缺点**：更新噪声大，收敛不稳定  
- **公式**：  

  $$
  \theta := \theta - \eta \cdot \nabla_\theta L(x_i, y_i, \theta)
  $$
### ✅ 优点：

- 每步计算非常快；
- 可以进行**在线学习**；
- 有助于逃出局部极小值（在非凸优化中）。

### ❌ 缺点：

- 收敛过程噪声大，曲折；
- 收敛不稳定，需要良好的学习率调节策略。
---

## 小批量梯度下降（Mini-Batch Gradient Descent）

结合了批量和随机的优点。每次迭代使用一个**小批量样本**（如32、64）更新参数。

- **优点**：计算效率高，梯度估计较稳定  
- **应用最广泛的梯度下降方法**  
- **公式**：  
  $$
\nabla J(\theta) = \frac{1}{|B|} \sum_{i \in B} \nabla_\theta \mathcal{L}(\theta; x^{(i)}, y^{(i)})
$$
$$
  \theta := \theta - \eta \nabla J(\theta)
  $$
### ✅ 优点：

- 综合了批量和随机的优点；
- 计算效率高，能利用向量化与 GPU 加速；
- 收敛过程平稳但仍有一定随机性，有助于跳出局部最小值。

### ❌ 缺点：

- 训练结果对 batch size 敏感；
- 参数调优复杂性增加（涉及 batch size、learning rate 等）。

---

## 动量法（Momentum）

为了解决梯度方向震荡问题，引入“动量”的概念。当前更新方向参考了**过去的梯度积累趋势**。

- **更新公式**：
  $$
  v_t = \gamma v_{t-1} + \eta \nabla_\theta L(\theta)
  $$
  $$
  \theta := \theta - v_t
  $$
- $\gamma$：动量系数，常设为 0.9

---

## 自适应梯度法（AdaGrad）

对每个参数使用**不同的学习率**，根据历史梯度自动调整。

- **优点**：适合稀疏特征的优化（如NLP中的词向量）  
- **缺点**：学习率单调下降，后期过小导致收敛困难  
- **公式**：  
  $$
  \theta_j := \theta_j - \frac{\eta}{\sqrt{G_{jj}} + \epsilon} \cdot \nabla_\theta L
  $$  
  其中 $G$ 是梯度平方累加矩阵

---

## RMSprop

RMSprop 改进了 AdaGrad，使用**指数加权平均**来控制学习率衰减。

- **优点**：解决了 AdaGrad 衰减过快问题，适合非凸优化  
- **更新公式**：
  $$
  E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma) g_t^2
  $$
  $$
  \theta := \theta - \frac{\eta}{\sqrt{E[g^2]_t} + \epsilon} \cdot g_t
  $$

---

## Adam优化器（Adam Optimizer）

综合了 Momentum 和 RMSprop 的优点，**既有动量，也有自适应学习率**。

- **广泛应用于深度学习训练**
- **更新公式**：
  $$
  m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
  $$
  $$
  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
  $$
  $$
  \hat{m}_t = \frac{m_t}{1 - \beta_1^t},\quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
  $$
  $$
  \theta := \theta - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
  $$

---

## 学习率调度（Learning Rate Scheduling）

训练过程中动态调整学习率，避免在训练前期震荡和后期过慢收敛。

### 常见策略：
- **Step decay**：每隔一定 epoch 降低学习率  
- **Exponential decay**：学习率以指数衰减  
- **Cosine annealing**：余弦函数方式调节，常用于 Warmup  
- **Cyclical Learning Rate**：在一定范围内周期性波动
