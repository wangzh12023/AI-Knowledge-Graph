

### 📌 一、期望（Expectation）

---

#### 🧮 1. 离散型随机变量

设 $X$ 是一个离散型随机变量，其概率质量函数为 $p(x)$，则期望定义为：

$$
\mathbb{E}[X] = \sum_x x \cdot p(x)
$$

#### 📈 2. 连续型随机变量

设 $X$ 是连续型随机变量，密度函数为 $f(x)$，则期望定义为：

$$
\mathbb{E}[X] = \int_{-\infty}^{+\infty} x \cdot f(x)\, dx
$$


#### 🔁 3. 期望的性质

- 线性性：
  $$
  \mathbb{E}[aX + b] = a\mathbb{E}[X] + b
  $$
- 若 $X \le Y$，则 $\mathbb{E}[X] \le \mathbb{E}[Y]$
- 常数 $c$ 的期望为 $c$：$\mathbb{E}[c] = c$

#### 📊 4. 条件期望（Conditional Expectation）

若 $X, Y$ 有联合分布，定义条件期望：

$$
\mathbb{E}[X \mid Y = y] = 
\begin{cases}
\sum_x x \cdot P(X = x \mid Y = y) & \text{离散} \\
\int_{-\infty}^{+\infty} x \cdot f_{X|Y}(x \mid y)\, dx & \text{连续}
\end{cases}
$$

#### 🌍 5. 全期望公式（Law of Total Expectation）

$$
\mathbb{E}[X] = \mathbb{E}_Y[\mathbb{E}[X \mid Y]] = \sum_y \mathbb{E}[X \mid Y = y] P(Y = y)
$$

或积分形式：

$$
\mathbb{E}[X] = \int \mathbb{E}[X \mid Y = y] f_Y(y)\, dy
$$

### 📌 二、方差（Variance）

#### 🧮 1. 定义

- **方差** 衡量随机变量对其期望的离散程度：
  $$
  \mathrm{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2]
  $$

- 等价形式：
  $$
  \mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
  $$

#### 🔁 2. 方差的性质

- 常数的方差为 0：$\mathrm{Var}(c) = 0$
- 缩放影响平方倍数：
  $$
  \mathrm{Var}(aX + b) = a^2 \mathrm{Var}(X)
  $$
- 两个独立变量之和的方差：
  $$
  \mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)
  $$

#### 🌍 3. 全方差公式（Law of Total Variance）

$$
\mathrm{Var}(X) = \mathbb{E}[\mathrm{Var}(X \mid Y)] + \mathrm{Var}(\mathbb{E}[X \mid Y])
$$

### 📌 三、联合分布下的期望与协方差

---

#### 📦 1. 联合期望

对于两个随机变量 $X$ 和 $Y$，其联合期望定义为：

$$
\mathbb{E}[XY] = 
\begin{cases}
\sum_x \sum_y xy \cdot P(X = x, Y = y) & \text{离散} \\
\int \int xy \cdot f_{X,Y}(x, y)\, dx\, dy & \text{连续}
\end{cases}
$$

---

#### 🔗 2. 协方差（Covariance）

##### ✅ 定义

协方差度量的是两个随机变量 $X, Y$ **同时偏离其均值的趋势**。

$$
\mathrm{Cov}(X, Y) = \mathbb{E}\left[(X - \mathbb{E}[X]) (Y - \mathbb{E}[Y])\right]
$$

或等价形式：

$$
\mathrm{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X] \cdot \mathbb{E}[Y]
$$

---

##### 🎯 几何意义

- 若 $X$ 大于其均值时，$Y$ 也倾向于大于其均值 → 协方差为正（正相关）
- 若 $X$ 增加时，$Y$ 倾向于减小 → 协方差为负（负相关）
- 若二者独立，通常 $\mathrm{Cov}(X, Y) = 0$

---

##### 📏 单位/数量级问题

协方差单位依赖于 $X$ 和 $Y$ 的量纲，例如：
- 身高(cm) 与体重(kg) → 协方差单位为 cm·kg
- 不适合直接比较不同特征之间的协方差

---

##### 📌 计算形式

###### 离散型：

$$
\mathrm{Cov}(X, Y) = \sum_x \sum_y (x - \mu_X)(y - \mu_Y) \cdot P(X = x, Y = y)
$$

###### 连续型：

$$
\mathrm{Cov}(X, Y) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} (x - \mu_X)(y - \mu_Y) \cdot f_{X,Y}(x, y)\, dx\, dy
$$

---

##### 🧮 特殊情况

- $\mathrm{Cov}(X, X) = \mathrm{Var}(X)$
- 若 $a, b$ 为常数：

$$
\mathrm{Cov}(aX + b, Y) = a \cdot \mathrm{Cov}(X, Y)
$$

- 若 $X, Y$ 独立，则：
  $$
  \mathrm{Cov}(X, Y) = 0
  $$
  但反之不一定成立（协方差为 0 ≠ 独立）

---

#### 📐 3. 相关系数（Correlation Coefficient）

##### ✅ 定义

相关系数是协方差的**标准化版本**，度量两个变量间的**线性相关程度**。

定义为：

$$
\rho_{X, Y} = \frac{\mathrm{Cov}(X, Y)}{\sigma_X \cdot \sigma_Y}
= \frac{\mathbb{E}[(X - \mu_X)(Y - \mu_Y)]}{\sqrt{\mathrm{Var}(X)} \cdot \sqrt{\mathrm{Var}(Y)}}
$$

---

##### 📐 值域与意义

- $\rho \in [-1, 1]$
- $\rho = 1$：完全正线性关系，$Y = aX + b, a > 0$
- $\rho = -1$：完全负线性关系，$Y = aX + b, a < 0$
- $\rho = 0$：**线性无关**，但不代表独立！

---

##### 🎯 几何/向量角度理解

- 把随机变量看作以期望为中心的向量
- $\rho_{X, Y} = \cos(\theta)$，其中 $\theta$ 是变量偏差向量间的夹角

---

##### 📊 与协方差的关系

| 属性 | 协方差 $\mathrm{Cov}(X, Y)$ | 相关系数 $\rho_{X,Y}$ |
|------|--------------------------|--------------------------|
| 描述 | 偏移趋势 | 线性强度 |
| 有无单位 | 有单位 | 无单位（归一化） |
| 值域 | $(-\infty, \infty)$ | $[-1, 1]$ |
| 易比较性 | 不易 | 可跨变量比较 |
| 取值 0 时意义 | 无线性趋势 | 无线性相关性 |

---


##### 🔁 协方差矩阵（Covariance Matrix）

对向量随机变量 $X = (X_1, X_2, ..., X_n)$，协方差矩阵定义为：

$$
\Sigma_{ij} = \mathrm{Cov}(X_i, X_j)
$$

- 对称矩阵，半正定
- 用于多变量统计分析、PCA、多元正态分布建模

---

##### 🐍 Python 示例：

```python
import numpy as np

X = np.array([1, 2, 3])
Y = np.array([2, 4, 6])

cov = np.cov(X, Y, ddof=0)[0, 1]
corr = np.corrcoef(X, Y)[0, 1]

print("协方差:", cov)
print("相关系数:", corr)
```

---

### 📌 四、高阶矩与中心矩（Moments）

- **k 阶矩**：$\mathbb{E}[X^k]$
- **k 阶中心矩**：$\mathbb{E}[(X - \mathbb{E}[X])^k]$
  - 二阶中心矩是方差
  - 三阶中心矩衡量偏度（Skewness）
  - 四阶中心矩衡量峰度（Kurtosis）

---

### ⚠️ 五、常见注意事项

- $\mathrm{Var}(X + Y) = \mathrm{Var}(X) + \mathrm{Var}(Y)$ **仅在 $X,Y$ 独立时成立**
- 期望是线性的，方差不是
- 方差是对称的，协方差可以为负
- 条件期望本身是一个函数 $\mathbb{E}[X \mid Y]$，可再取期望或方差
