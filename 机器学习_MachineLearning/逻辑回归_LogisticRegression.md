

## 逻辑回归（Logistic Regression）

逻辑回归是一种广泛应用于二分类问题的线性模型，它通过将线性回归结果映射为概率值，来判断样本属于某一类别的可能性。

---

## 一、模型定义

逻辑回归模型形式为：

$$
f(x) = \sigma(\boldsymbol{w}^T \boldsymbol{x} + b)
$$

其中 $\sigma(\cdot)$ 是 sigmoid 函数：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

因此，模型输出为：

$$
P(y=1 \mid x) = \frac{1}{1 + e^{-(\boldsymbol{w}^T x + b)}}
$$

属于类别 0 的概率为：

$$
P(y=0 \mid x) = 1 - P(y=1 \mid x)
$$

---

## 二、判别准则

### 使用Bayes 分类器

Bayes 分类器选取**后验概率最大**的类别：

$$
\hat{y} = \arg\max_{y \in \{0, 1\}} P(y \mid x)
$$

在逻辑回归中即：

$$
\hat{y} =
\begin{cases}
1, & \text{if } \sigma(\boldsymbol{w}^T x + b) \geq 0.5 \\
0, & \text{otherwise}
\end{cases}
$$

等价于：

$$
\hat{y} =
\begin{cases}
1, & \text{if } \boldsymbol{w}^T x + b \geq 0 \\
0, & \text{otherwise}
\end{cases}
$$


决策边界对应于：

$$
\boldsymbol{w}^T x + b = 0
$$

它是一个线性边界（超平面）。

---

## 三、损失函数（对数似然）

用[MLE](./../数学基础_MathBasics/概率论与数理统计_ProbabilityAndStatistics/数理统计推断_StatisticalInference#2.2极大似然估计（MLE）)来定义损失函数

为何使用对数似然而非原始似然
1. **简化乘积为加法**，便于求导和数值优化；
2. 对数是严格递增函数，不影响最优解；
3. 防止数值下溢（乘积易变成 0）；
4. 方便分析、梯度计算与收敛性证明。

逻辑回归通过最大化似然函数（Maximum Likelihood）来估计参数。对数似然函数为：

$$
\log L(\boldsymbol{w}, b) = \sum_{i=1}^m \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)}) \right]
$$

负对数似然即为交叉熵损失函数：

$$
J(\boldsymbol{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)}) \right]
$$


- 解释：

逻辑回归假设每个样本的标签 $y^{(i)} \in \{0, 1\}$ 是一个**伯努利分布变量**，其分布参数为：

$$
P(y=1 \mid x^{(i)}) = \hat{y}^{(i)} = \sigma(\boldsymbol{w}^T x^{(i)} + b)
$$

则：

$$
P(y^{(i)} \mid x^{(i)}; \boldsymbol{w}) =
[\hat{y}^{(i)}]^{y^{(i)}} [1 - \hat{y}^{(i)}]^{1 - y^{(i)}}
$$

训练样本共有 $m$ 个，记数据集为：

$$
\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^m
$$

则整个训练集的**似然函数**为：

$$
L(\boldsymbol{w}, b) = \prod_{i=1}^m [\hat{y}^{(i)}]^{y^{(i)}} [1 - \hat{y}^{(i)}]^{1 - y^{(i)}}
$$
对数似然函数（Log-Likelihood）

由于连乘求导不便，取对数：

$$
\log L(\boldsymbol{w}, b) = \sum_{i=1}^m \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)}) \right]
$$

在训练中我们最小化的是负对数似然：

$$
J(\boldsymbol{w}, b) = - \log L(\boldsymbol{w}, b)
= - \sum_{i=1}^m \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)}) \right]
$$

这正是我们常见的 **二分类交叉熵损失函数**。

> **逻辑回归训练的本质：最大似然估计**

![[Pasted image 20250415094210.png]]


![[Pasted image 20250415094200.png]]


![[Pasted image 20250415094233.png]]


---

## 四、梯度计算与优化

对损失函数求导得到参数的梯度：

- 权重梯度：

$$
\frac{\partial J}{\partial \boldsymbol{w}} = \frac{1}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \right) x^{(i)}
$$

- 偏置梯度：

$$
\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m \left( \hat{y}^{(i)} - y^{(i)} \right)
$$

采用梯度下降更新参数：

$$
\boldsymbol{w} := \boldsymbol{w} - \eta \cdot \frac{\partial J}{\partial \boldsymbol{w}}, \quad
b := b - \eta \cdot \frac{\partial J}{\partial b}
$$

---

## 五、正则化（防止过拟合）

引入[正则化](正则化_Regularization)项
### L2 正则化（Ridge）：

$$
J(\boldsymbol{w}, b) = \text{CrossEntropy} + \frac{\lambda}{2} \|\boldsymbol{w}\|^2
$$

### L1 正则化（Lasso）：

$$
J(\boldsymbol{w}, b) = \text{CrossEntropy} + \lambda \|\boldsymbol{w}\|_1
$$

---

## 六、多分类扩展（Softmax 回归）

对于 $K$ 类分类任务，可扩展为 softmax 回归（多项逻辑回归）：

$$
P(y=k \mid x) = \frac{e^{\boldsymbol{w}_k^T x}}{\sum_{j=1}^K e^{\boldsymbol{w}_j^T x}}
$$

对应的损失函数为多类别交叉熵：

$$
J = -\sum_{i=1}^m \sum_{k=1}^K \mathbf{1}\{y^{(i)} = k\} \log P(y^{(i)} = k \mid x^{(i)})
$$

---

## 七、优缺点总结

### ✅ 优点：

- 模型简单，易于实现；
- 输出为概率值，可解释性强；
- 对特征线性可分任务效果好；
- 可引入正则项防止过拟合。

### ❌ 缺点：

- 只能拟合线性边界；
- 对异常值敏感；
- 特征之间需线性独立（共线性会影响训练）。


