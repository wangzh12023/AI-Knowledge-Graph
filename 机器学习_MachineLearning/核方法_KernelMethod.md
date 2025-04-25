

## 核方法与核回归（Kernel Methods and Kernel Regression）

核方法是通过将数据映射到高维空间，使得原本线性不可分的问题在高维空间中变得线性可分，从而构造更强的非线性模型。

---

## 一、核方法的基本思想

核方法的核心在于“**用核函数代替显式特征映射**”。

设有映射函数：

$$
\phi: \mathbb{R}^n \rightarrow \mathbb{H}
$$

我们希望在特征空间 $\mathbb{H}$ 中计算内积：

$$
\langle \phi(x), \phi(x') \rangle
$$

但实际不构造 $\phi(x)$，而是用**核函数** $K(x, x')$ 直接表示该内积：

$$
K(x, x') = \langle \phi(x), \phi(x') \rangle
$$

> **核技巧（Kernel Trick）**：避免显式高维映射，仍然可进行非线性学习。

---

## 二、常见核函数

| 核函数      | 表达式                                                           | 特点         |
| -------- | ------------------------------------------------------------- | ---------- |
| 线性核      | $K(x, x') = x^T x'$                                           | 不映射，等价线性模型 |
| 多项式核     | $K(x, x') = (x^T x' + c)^d$                                   | 控制多项式非线性程度 |
| 高斯核（RBF） | $K(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right)$ | 无限维映射，强非线性 |
| Sigmoid核 | $K(x, x') = \tanh(\alpha x^T x' + \beta)$                     | 类似神经网络激活   |

---

## 三、核回归（KernelRegression）

核回归是一种非参数回归方法，使用核函数来进行加权平均预测。它的基本思想是：

> 对于一个新样本 $x$，根据其与训练集中各点的相似性（核函数值）对 $y$ 进行加权平均预测。

---

### 3.1 Nadaraya-Watson 核回归

设训练数据为 $\{(x_i, y_i)\}_{i=1}^n$，则预测函数为：

$$
\hat{y}(x) = \frac{\sum_{i=1}^{n} K(x, x_i) y_i}{\sum_{i=1}^{n} K(x, x_i)}
$$

其中 $K(x, x_i)$ 是核函数（常用 RBF 核）。

---

### 3.2 优点与缺点

✅ 优点：

- 非参数建模，适应能力强；
- 不需要拟合模型参数；
- 本质是局部加权平均，结果平滑。

❌ 缺点：

- 计算复杂度高：$O(n)$ 计算每个预测；
- 对带宽参数（如 $\sigma$）敏感；
- 在高维空间中面临“维度灾难”。

---

## 四、核岭回归（Kernel Ridge Regression）

将岭回归与核技巧结合，形成非线性回归模型：

目标函数：

$$
\min_{\boldsymbol{\alpha}} \left\| \boldsymbol{y} - K \boldsymbol{\alpha} \right\|^2 + \lambda \boldsymbol{\alpha}^T K \boldsymbol{\alpha}
$$

其中：

- $K \in \mathbb{R}^{n \times n}$ 是核矩阵，$K_{ij} = K(x_i, x_j)$；
- 解为闭式：

$$
\boldsymbol{\alpha} = (K + \lambda I)^{-1} \boldsymbol{y}
$$

最终预测为：

$$
\hat{y}(x) = \sum_{i=1}^n \alpha_i K(x, x_i)
$$

---

## 五、核方法的应用举例

- 支持向量机（SVM）
- 高斯过程（Gaussian Process）
- 核主成分分析（Kernel PCA）
- 核岭回归（KRR）
- 核逻辑回归（Kernel Logistic Regression）

---

## 六、核函数的合法性条件

**Mercer 定理**：函数 $K(x, x')$ 是合法核函数，当且仅当对任意样本集合，生成的核矩阵 $K$ 是对称、半正定的。

---

## 七、总结

| 内容 | 说明 |
|------|------|
| 核方法 | 用核函数代替高维映射，构建非线性模型 |
| 核回归 | 基于核加权平均的非参数回归方法 |
| 核技巧 | 避免显式构造 $\phi(x)$，直接操作核函数 |
| 应用场景 | 支持向量机、岭回归、PCA 等 |

---

## 参考资料

- 《统计学习方法》李航
- 《机器学习》周志华
- PRML – Bishop
- scikit-learn kernel methods 文档
